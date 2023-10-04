export enum ServerStatus {
  STOPPED = 'STOPPED',
  STARTING = 'STARTING',
  STARTED = 'STARTED',
}

export enum ServerStartingStage {
  DETECT_SYSTEM_PYTHON,
  CREATE_VENV,
  CHECK_VENV_ACTIVATION,
  UPGRADE_PIP,
  INSTALL_REQUIREMENTS,
  START_SERVER,
}

export interface ServerState {
  status: ServerStatus;
  stage: ServerStartingStage | null;
}

export const INITIAL_SERVER_STATE: ServerState = {
  status: ServerStatus.STOPPED,
  stage: null,
};
