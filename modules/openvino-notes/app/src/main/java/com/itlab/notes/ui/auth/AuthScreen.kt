package com.itlab.notes.ui.auth

import android.app.Activity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.Email
import androidx.compose.material.icons.rounded.Visibility
import androidx.compose.material.icons.rounded.VisibilityOff
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.google.android.gms.auth.api.signin.GoogleSignIn
import com.google.android.gms.auth.api.signin.GoogleSignInOptions
import com.google.android.gms.auth.api.signin.GoogleSignInStatusCodes
import com.google.android.gms.common.api.ApiException
import com.itlab.notes.R
import org.koin.androidx.compose.koinViewModel

@Composable
fun authScreen(viewModel: AuthViewModel = koinViewModel()) {
    val state by viewModel.uiState.collectAsState()
    val context = LocalContext.current

    val webClientId = stringResource(R.string.default_web_client_id)
    val googleSignInEnabled = webClientId.isNotBlank()

    val googleSignInClient =
        remember(context, webClientId) {
            val options =
                GoogleSignInOptions
                    .Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
                    .requestIdToken(webClientId)
                    .requestEmail()
                    .build()
            GoogleSignIn.getClient(context, options)
        }

    val googleLauncher =
        rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            val data = result.data
            if (data == null) {
                if (result.resultCode != Activity.RESULT_CANCELED) {
                    viewModel.reportError("Google sign-in returned no data.")
                } else {
                    viewModel.clearError()
                }
                return@rememberLauncherForActivityResult
            }
            try {
                val account =
                    GoogleSignIn
                        .getSignedInAccountFromIntent(data)
                        .getResult(ApiException::class.java)
                val token = account.idToken
                if (token.isNullOrBlank()) {
                    viewModel.reportError(
                        "Google sign-in did not return a token. Check Web Client ID in Firebase.",
                    )
                } else {
                    viewModel.signInWithGoogle(token)
                }
            } catch (error: ApiException) {
                if (error.statusCode == GoogleSignInStatusCodes.SIGN_IN_CANCELLED) {
                    viewModel.clearError()
                } else {
                    viewModel.reportError(mapGoogleSignInError(error))
                }
            }
        }

    val googleUnavailableMessage =
        "Google sign-in requires a Web Client ID in Firebase (see default_web_client_id)."
    val launchGoogleSignIn = {
        if (!googleSignInEnabled) {
            viewModel.reportError(googleUnavailableMessage)
        } else {
            viewModel.clearError()
            viewModel.clearSuccess()
            googleLauncher.launch(googleSignInClient.signInIntent)
        }
    }

    Scaffold { padding ->
        Box(
            modifier =
                Modifier
                    .fillMaxSize()
                    .padding(padding),
        ) {
            key(state.step) {
                when (state.step) {
                    AuthScreenStep.ChooseMethod ->
                        authMethodChoiceContent(
                            isLoading = state.isLoading,
                            googleSignInEnabled = googleSignInEnabled,
                            onGoogleClick = launchGoogleSignIn,
                            onEmailClick = { viewModel.openEmailStep() },
                            errorMessage = state.errorMessage,
                        )
                    AuthScreenStep.Email ->
                        authEmailContent(
                            state = state,
                            onBackFromEmail = { viewModel.backFromEmailStep() },
                            onSignIn = viewModel::signInWithEmail,
                            onSignUp = viewModel::signUpWithEmail,
                            onSwitchToSignUp = { viewModel.switchToSignUpMode() },
                            onSwitchToSignIn = { viewModel.switchToSignInMode() },
                            onClearError = { viewModel.clearError() },
                            onClearSuccess = { viewModel.clearSuccess() },
                        )
                }
            }
        }
    }
}

@Composable
private fun authMethodChoiceContent(
    isLoading: Boolean,
    googleSignInEnabled: Boolean,
    onGoogleClick: () -> Unit,
    onEmailClick: () -> Unit,
    errorMessage: String?,
) {
    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .padding(horizontal = 24.dp)
                .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "Notes",
            style = MaterialTheme.typography.headlineMedium,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Choose how you want to sign in",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(40.dp))

        authMessageBlock(errorMessage = errorMessage, successMessage = null)

        OutlinedButton(
            onClick = onGoogleClick,
            modifier =
                Modifier
                    .fillMaxWidth()
                    .height(48.dp),
            enabled = !isLoading && googleSignInEnabled,
        ) {
            if (isLoading) {
                authButtonLoadingIndicator(
                    color = MaterialTheme.colorScheme.primary,
                )
            } else {
                Icon(
                    painter = painterResource(R.drawable.ic_google),
                    contentDescription = null,
                    modifier = Modifier.authMethodIcon(),
                    tint = Color.Unspecified,
                )
                Text("Continue with Google")
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        OutlinedButton(
            onClick = onEmailClick,
            modifier =
                Modifier
                    .fillMaxWidth()
                    .height(48.dp),
            enabled = !isLoading,
        ) {
            Icon(
                imageVector = Icons.Rounded.Email,
                contentDescription = null,
                modifier = Modifier.authMethodIcon(),
            )
            Text("Sign in with Email")
        }

        if (!googleSignInEnabled) {
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "Google sign-in requires a Web Client ID in Firebase (see default_web_client_id).",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun authEmailContent(
    state: AuthUiState,
    onBackFromEmail: () -> Unit,
    onSignIn: (String, String) -> Unit,
    onSignUp: (String, String) -> Unit,
    onSwitchToSignUp: () -> Unit,
    onSwitchToSignIn: () -> Unit,
    onClearError: () -> Unit,
    onClearSuccess: () -> Unit,
) {
    BackHandler(onBack = onBackFromEmail)

    var email by rememberSaveable { mutableStateOf("") }
    var password by rememberSaveable { mutableStateOf("") }
    var passwordVisible by rememberSaveable { mutableStateOf(false) }

    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = {
                Text(
                    text =
                        if (state.isSignUpMode) {
                            "Create account"
                        } else {
                            "Sign in"
                        },
                )
            },
            navigationIcon = {
                IconButton(onClick = onBackFromEmail, enabled = !state.isLoading) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Rounded.ArrowBack,
                        contentDescription = "Back",
                    )
                }
            },
            colors =
                TopAppBarDefaults.topAppBarColors(
                    containerColor = Color.Transparent,
                ),
        )

        Column(
            modifier =
                Modifier
                    .fillMaxSize()
                    .padding(horizontal = 24.dp)
                    .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            OutlinedTextField(
                value = email,
                onValueChange = {
                    email = it
                    onClearError()
                    onClearSuccess()
                },
                shape = MaterialTheme.shapes.medium,
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Email") },
                singleLine = true,
                keyboardOptions =
                    KeyboardOptions(
                        keyboardType = KeyboardType.Email,
                        imeAction = ImeAction.Next,
                    ),
                enabled = !state.isLoading,
            )
            Spacer(modifier = Modifier.height(12.dp))
            OutlinedTextField(
                value = password,
                onValueChange = {
                    password = it
                    onClearError()
                    onClearSuccess()
                },
                shape = MaterialTheme.shapes.medium,
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Password") },
                singleLine = true,
                visualTransformation =
                    if (passwordVisible) {
                        VisualTransformation.None
                    } else {
                        PasswordVisualTransformation()
                    },
                trailingIcon = {
                    IconButton(onClick = { passwordVisible = !passwordVisible }) {
                        Icon(
                            imageVector =
                                if (passwordVisible) {
                                    Icons.Rounded.VisibilityOff
                                } else {
                                    Icons.Rounded.Visibility
                                },
                            contentDescription =
                                if (passwordVisible) {
                                    "Hide password"
                                } else {
                                    "Show password"
                                },
                        )
                    }
                },
                keyboardOptions =
                    KeyboardOptions(
                        keyboardType = KeyboardType.Password,
                        imeAction = ImeAction.Done,
                    ),
                enabled = !state.isLoading,
            )

            Spacer(modifier = Modifier.height(12.dp))
            authMessageBlock(
                errorMessage = state.errorMessage,
                successMessage = state.successMessage,
            )

            Spacer(modifier = Modifier.height(24.dp))

            Button(
                onClick = {
                    if (state.isSignUpMode) {
                        onSignUp(email, password)
                    } else {
                        onSignIn(email, password)
                    }
                },
                modifier =
                    Modifier
                        .fillMaxWidth()
                        .height(48.dp),
                enabled = !state.isLoading,
            ) {
                if (state.isLoading) {
                    authButtonLoadingIndicator(
                        color = MaterialTheme.colorScheme.onPrimary,
                    )
                } else {
                    Text(
                        text =
                            if (state.isSignUpMode) {
                                "Create account"
                            } else {
                                "Sign in"
                            },
                    )
                }
            }

            TextButton(
                onClick = if (state.isSignUpMode) onSwitchToSignIn else onSwitchToSignUp,
                enabled = !state.isLoading,
            ) {
                Text(
                    text =
                        if (state.isSignUpMode) {
                            "Already have an account? Sign in"
                        } else {
                            "Need an account? Create one"
                        },
                )
            }
        }
    }
}

private fun Modifier.authMethodIcon(): Modifier =
    size(30.dp)
        .padding(end = 12.dp)

@Composable
private fun authButtonLoadingIndicator(color: Color = LocalContentColor.current) {
    CircularProgressIndicator(
        modifier = Modifier.size(22.dp),
        strokeWidth = 2.dp,
        color = color,
    )
}

@Composable
private fun authMessageBlock(
    errorMessage: String?,
    successMessage: String?,
) {
    when {
        !errorMessage.isNullOrBlank() ->
            Text(
                text = errorMessage,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
            )
        !successMessage.isNullOrBlank() ->
            Text(
                text = successMessage,
                color = MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.bodySmall,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
            )
    }
}

private fun mapGoogleSignInError(error: ApiException): String =
    when (error.statusCode) {
        GoogleSignInStatusCodes.DEVELOPER_ERROR ->
            "Google Sign-In configuration error. Add SHA-1 fingerprint in Firebase Console."
        else -> error.message ?: "Google sign-in failed (code ${error.statusCode})."
    }
