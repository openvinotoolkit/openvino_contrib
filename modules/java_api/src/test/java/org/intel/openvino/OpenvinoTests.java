package org.intel.openvino;

import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class OpenvinoTests extends OVTest {

    private Model model;

    @Before
    public void init() {
        Core core = new Core();
        model = core.read_model(modelXml);
    }

    @Test
    public void testSerialize() throws IOException {
        File tmp = Files.createTempDirectory("ovtest").toFile();
        File xmlPath = Paths.get(tmp.getAbsolutePath(), "saved_model.xml").toFile();
        File binPath = Paths.get(tmp.getAbsolutePath(), "saved_model.bin").toFile();
        xmlPath.deleteOnExit();
        binPath.deleteOnExit();
        tmp.deleteOnExit();

        Openvino.serialize(model, xmlPath.getAbsolutePath(), binPath.getAbsolutePath());
        assertTrue(xmlPath.exists() && binPath.exists());
    }

    @Test
    public void testSaveModel() throws IOException {
        File tmp = Files.createTempDirectory("ovSaveModelTest").toFile();
        File xmlPath = Paths.get(tmp.getAbsolutePath(), "saved_model.xml").toFile();
        File binPath = Paths.get(tmp.getAbsolutePath(), "saved_model.bin").toFile();
        xmlPath.deleteOnExit();
        binPath.deleteOnExit();
        tmp.deleteOnExit();

        Openvino.save_model(model, xmlPath.getAbsolutePath(), false);
        assertTrue(xmlPath.exists() && binPath.exists());
    }
}
