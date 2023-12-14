import org.intel.openvino.Core;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.logging.ConsoleHandler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

/*
This sample queries all available OpenVINO Runtime Devices and prints
their supported metrics and plugin configuration parameters using Query
Device API feature. The application prints all available devices with
their supported metrics and default values for configuration parameters.

The sample takes no command-line parameters.
*/
public class Main {

    private static Logger logger;

    static {
        logger = Logger.getLogger(Main.class.getName());
        logger.setUseParentHandlers(false);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setFormatter(
                new SimpleFormatter() {
                    private static final String format = "[%1$s] %2$s%n";

                    @Override
                    public synchronized String format(LogRecord lr) {
                        return String.format(
                                format, lr.getLevel().getLocalizedName(), lr.getMessage());
                    }
                });
        logger.addHandler(handler);
    }

    public static void main(String[] args) throws IOException {
        List<String> excludedProperties =
                Arrays.asList("SUPPORTED_METRICS", "SUPPORTED_CONFIG_KEYS", "SUPPORTED_PROPERTIES");

        Core core = new Core();

        logger.info("Available devices:");
        for (String device : core.get_available_devices()) {
            logger.info(String.format("%s:", device));
            logger.info("\tSUPPORTED_PROPERTIES:");

            for (String propertyKey : core.get_property(device, "SUPPORTED_PROPERTIES").asList()) {
                if (!excludedProperties.contains(propertyKey)) {
                    String propertyVal;
                    try {
                        propertyVal = core.get_property(device, propertyKey).asString();
                    } catch (Exception e) {
                        propertyVal = "UNSUPPORTED TYPE";
                    }
                    logger.info(String.format("\t\t%s: %s", propertyKey, propertyVal));
                }
            }
        }
    }
}
