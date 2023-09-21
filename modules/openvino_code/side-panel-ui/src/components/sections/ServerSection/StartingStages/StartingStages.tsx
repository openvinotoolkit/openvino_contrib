import { ServerStartingStage } from '@shared/server-state';
import { ReactElement } from 'react';
import { VscodeIcon } from '../../../shared/VscodeIcon/VscodeIcon';
import './StartingStages.css';

const startingStages = [
  ServerStartingStage.DETECT_SYSTEM_PYTHON,
  ServerStartingStage.CREATE_VENV,
  ServerStartingStage.CHECK_VENV_ACTIVATION,
  ServerStartingStage.UPGRADE_PIP,
  ServerStartingStage.INSTALL_REQUIREMENTS,
  ServerStartingStage.START_SERVER,
];

const startingStagesLabelsMap = {
  [ServerStartingStage.DETECT_SYSTEM_PYTHON]: 'Detecting system Python',
  [ServerStartingStage.CREATE_VENV]: 'Creating virtual environment',
  [ServerStartingStage.CHECK_VENV_ACTIVATION]: 'Activating virtual environment',
  [ServerStartingStage.UPGRADE_PIP]: 'Upgrading pip',
  [ServerStartingStage.INSTALL_REQUIREMENTS]: 'Installing dependencies',
  [ServerStartingStage.START_SERVER]: 'Starting server',
};

interface StartingStageProps {
  stage: ServerStartingStage;
  icon: ReactElement<typeof VscodeIcon>;
}

const StartingStage = ({ stage, icon }: StartingStageProps): JSX.Element => {
  return (
    <span className="starting-stage-item">
      {icon}&nbsp;{startingStagesLabelsMap[stage]}
    </span>
  );
};

const getStageIcon = (
  itemStage: ServerStartingStage,
  currentStage: StartingStagesProps['currentStage']
): ReactElement<typeof VscodeIcon> => {
  if (currentStage === itemStage) {
    return <VscodeIcon iconName="loading" spin></VscodeIcon>;
  } else if (currentStage && currentStage > itemStage) {
    return <VscodeIcon iconName="check"></VscodeIcon>;
  } else {
    return <VscodeIcon iconName="debug-pause"></VscodeIcon>;
  }
};

interface StartingStagesProps {
  currentStage: ServerStartingStage | null;
}

export const StartingStages = ({ currentStage }: StartingStagesProps): JSX.Element => {
  return (
    <pre className="starting-stages">
      {startingStages.map((itemStage) => (
        <StartingStage key={itemStage} stage={itemStage} icon={getStageIcon(itemStage, currentStage)}></StartingStage>
      ))}
    </pre>
  );
};
