import { ConfigurationTarget, ExtensionContext, commands } from 'vscode';
import { COMMANDS, CONFIG_KEY, EXTENSION_PACKAGE, EXTENSION_DISPLAY_NAME } from '../constants';
import { CustomConfiguration } from '../configuration';
import { IExtensionComponent } from '../extension-component.interface';
import { extensionState } from '../state';

class SettingsService implements IExtensionComponent {
  activate(context: ExtensionContext): void {
    const settingsCommandDisposable = commands.registerCommand(
      COMMANDS.OPEN_SETTINGS,
      (key?: keyof CustomConfiguration) => {
        this.openSettings(key);
      }
    );

    const showLogCommandDisposable = commands.registerCommand(COMMANDS.SHOW_EXTENSION_LOG, () => {
      void commands.executeCommand(
        `workbench.action.output.show.${EXTENSION_PACKAGE.fullName}.${EXTENSION_DISPLAY_NAME}`
      );
    });

    context.subscriptions.push(settingsCommandDisposable, showLogCommandDisposable);
  }

  deactivate(): void {}

  openSettings(key?: keyof CustomConfiguration): void {
    const configKey = key ? [CONFIG_KEY, key].join('.') : CONFIG_KEY;
    void commands.executeCommand('workbench.action.openSettings', configKey);
  }

  updateSetting<K extends keyof CustomConfiguration>(key: K, value: CustomConfiguration[K]): void {
    // FIXME, TODO: model selection configuration update doesn't work if configuration is in .vscode/settings.json
    void extensionState.config.update(key, value, ConfigurationTarget.Global);
  }
}

export const settingsService = new SettingsService();
