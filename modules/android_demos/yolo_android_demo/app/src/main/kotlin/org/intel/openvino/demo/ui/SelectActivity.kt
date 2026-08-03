// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.ui

import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import org.intel.openvino.demo.R
import org.intel.openvino.demo.data.DownloadErrorKind
import org.intel.openvino.demo.data.DownloadState
import org.intel.openvino.demo.data.ManifestRepository
import org.intel.openvino.demo.data.ModelEntry
import org.intel.openvino.demo.data.ModelManifest

/**
 * Home screen (§6b). Loads the bundled manifest offline, lets the user pick YOLO version → size →
 * task (cascading spinners), shows the download size, and on Start downloads the model (progress +
 * cancel + no-internet dialog). On success it launches [CameraActivity] with the local ONNX path.
 */
class SelectActivity : AppCompatActivity() {

    private lateinit var manifest: ModelManifest
    private lateinit var vm: DownloadViewModel

    private lateinit var versionSpinner: Spinner
    private lateinit var sizeSpinner: Spinner
    private lateinit var taskSpinner: Spinner
    private lateinit var sizeLabel: TextView
    private lateinit var startButton: Button
    private lateinit var progressGroup: LinearLayout
    private lateinit var progressBar: ProgressBar
    private lateinit var progressLabel: TextView
    private lateinit var cancelButton: Button
    private lateinit var clearCacheButton: Button

    private var selected: ModelEntry? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_select)

        versionSpinner = findViewById(R.id.versionSpinner)
        sizeSpinner = findViewById(R.id.sizeSpinner)
        taskSpinner = findViewById(R.id.taskSpinner)
        sizeLabel = findViewById(R.id.sizeLabel)
        startButton = findViewById(R.id.startButton)
        progressGroup = findViewById(R.id.progressGroup)
        progressBar = findViewById(R.id.progressBar)
        progressLabel = findViewById(R.id.progressLabel)
        cancelButton = findViewById(R.id.cancelButton)
        clearCacheButton = findViewById(R.id.clearCacheButton)

        vm = ViewModelProvider(this)[DownloadViewModel::class.java]

        try {
            manifest = ManifestRepository(this).load()
        } catch (t: Throwable) {
            Toast.makeText(this, "Failed to load model manifest: ${t.message}", Toast.LENGTH_LONG).show()
            return
        }

        setupSpinners()
        observeDownload()

        startButton.setOnClickListener { startDownload() }
        cancelButton.setOnClickListener { vm.cancel() }
        clearCacheButton.setOnClickListener { onClearCache() }
    }

    override fun onResume() {
        super.onResume()
        // Re-read cache state every time the screen is shown (e.g. after returning from the camera
        // with a freshly downloaded model, or after clearing), so the labels never go stale.
        if (::manifest.isInitialized) {
            refreshSizeLabel()
            refreshClearCacheButton()
        }
    }

    /** Update the "Clear cache" button label with the current cache size (or disable if empty). */
    private fun refreshClearCacheButton() {
        val bytes = vm.cacheBytes()
        val n = vm.cacheCount()
        clearCacheButton.isEnabled = bytes > 0
        clearCacheButton.text = if (bytes > 0) {
            "${getString(R.string.clear_cache)} ($n · ${humanBytes(bytes)})"
        } else {
            getString(R.string.clear_cache)
        }
    }

    private fun onClearCache() {
        val bytes = vm.cacheBytes()
        if (bytes <= 0) {
            Toast.makeText(this, R.string.clear_cache_empty, Toast.LENGTH_SHORT).show(); return
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.clear_cache_title)
            .setMessage("This will delete ${vm.cacheCount()} downloaded model(s) (${humanBytes(bytes)}). They will be re-downloaded when selected.")
            .setPositiveButton(android.R.string.ok) { d, _ ->
                val freed = vm.clearCache()
                Toast.makeText(this, "Freed ${humanBytes(freed)}", Toast.LENGTH_SHORT).show()
                refreshClearCacheButton()
                refreshSizeLabel()
                d.dismiss()
            }
            .setNegativeButton(android.R.string.cancel) { d, _ -> d.dismiss() }
            .show()
    }

    private fun setupSpinners() {
        setSpinner(versionSpinner, manifest.versions.map { versionLabel(it) })
        versionSpinner.onItemSelectedListener = onSelect { refreshSizes(); refreshTasks(); refreshSizeLabel() }
        sizeSpinner.onItemSelectedListener = onSelect { refreshTasks(); refreshSizeLabel() }
        taskSpinner.onItemSelectedListener = onSelect { refreshSizeLabel() }
        refreshSizes(); refreshTasks(); refreshSizeLabel()
    }

    private fun currentVersion(): String = manifest.versions[versionSpinner.selectedItemPosition]

    private fun refreshSizes() {
        setSpinner(sizeSpinner, manifest.sizesFor(currentVersion()).map { ModelEntry.sizeLabel(it) })
    }

    private fun refreshTasks() {
        val sizes = manifest.sizesFor(currentVersion())
        val size = sizes.getOrElse(sizeSpinner.selectedItemPosition) { sizes.first() }
        setSpinner(taskSpinner, manifest.tasksFor(currentVersion(), size).map { it.replaceFirstChar { c -> c.uppercase() } })
    }

    private fun refreshSizeLabel() {
        val entry = resolveSelection() ?: return
        selected = entry
        val cached = vm.cachedPath(entry) != null
        sizeLabel.text = if (cached) {
            "${entry.displayName} — Ready (cached, no download)"
        } else {
            "${entry.displayName} — download ${humanBytes(entry.bytes)}"
        }
    }

    private fun resolveSelection(): ModelEntry? {
        val version = currentVersion()
        val sizes = manifest.sizesFor(version)
        val size = sizes.getOrElse(sizeSpinner.selectedItemPosition) { return null }
        val tasks = manifest.tasksFor(version, size)
        val task = tasks.getOrElse(taskSpinner.selectedItemPosition) { return null }
        return manifest.find(version, size, task)
    }

    private fun startDownload() {
        val entry = resolveSelection() ?: return
        selected = entry
        // Cache hit → go straight to camera.
        vm.cachedPath(entry)?.let { launchCamera(entry, it); return }
        if (!vm.hasInternet()) { showErrorDialog(DownloadErrorKind.NO_INTERNET, "No internet connection."); return }
        setDownloadingUi(true)
        vm.start(entry)
    }

    private fun observeDownload() {
        vm.state.observe(this) { state ->
            when (state) {
                is DownloadState.Idle -> setDownloadingUi(false)
                is DownloadState.Downloading -> {
                    setDownloadingUi(true)
                    progressBar.progress = state.percent
                    progressLabel.text =
                        "Downloading… ${state.percent}%  (${humanBytes(state.downloadedBytes)} / ${humanBytes(state.totalBytes)})"
                }
                is DownloadState.Done -> {
                    setDownloadingUi(false)
                    selected?.let { launchCamera(it, state.path) }
                    vm.reset()
                }
                is DownloadState.Error -> {
                    setDownloadingUi(false)
                    if (state.kind != DownloadErrorKind.CANCELLED) {
                        showErrorDialog(state.kind, state.message)
                    }
                    vm.reset()
                }
            }
        }
    }

    private fun setDownloadingUi(downloading: Boolean) {
        progressGroup.visibility = if (downloading) View.VISIBLE else View.GONE
        startButton.isEnabled = !downloading
        versionSpinner.isEnabled = !downloading
        sizeSpinner.isEnabled = !downloading
        taskSpinner.isEnabled = !downloading
    }

    private fun showErrorDialog(kind: DownloadErrorKind, message: String) {
        val title = if (kind == DownloadErrorKind.NO_INTERNET)
            getString(R.string.no_internet_title) else getString(R.string.download_failed_title)
        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setCancelable(true)
            .setPositiveButton(R.string.retry) { d, _ -> d.dismiss(); startDownload() }
            .setNegativeButton(android.R.string.cancel) { d, _ -> d.dismiss() }
            .show()
    }

    private fun launchCamera(entry: ModelEntry, path: String) {
        startActivity(
            Intent(this, CameraActivity::class.java).apply {
                putExtra(CameraActivity.EXTRA_MODEL_ID, entry.id)
                putExtra(CameraActivity.EXTRA_ONNX_PATH, path)
            }
        )
    }

    // --- small helpers ---

    private fun setSpinner(spinner: Spinner, items: List<String>) {
        spinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, items)
    }

    private fun onSelect(action: () -> Unit) = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) = action()
        override fun onNothingSelected(p: AdapterView<*>?) {}
    }

    private fun versionLabel(v: String): String = ModelEntry.versionLabel(v)

    private fun humanBytes(b: Long): String {
        if (b <= 0) return "?"
        val mb = b / (1024.0 * 1024.0)
        return "%.1f MB".format(mb)
    }
}
