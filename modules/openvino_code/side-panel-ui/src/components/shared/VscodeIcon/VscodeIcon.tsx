import './VscodeIcon.css';

interface VscodeIconProps {
  iconName: string;
  spin?: boolean;
}

export const VscodeIcon = ({ iconName, spin }: VscodeIconProps): JSX.Element => {
  const classNames = ['codicon', `codicon-${iconName}`];
  if (spin) {
    classNames.push('codicon-spin');
  }
  return <div className={classNames.join(' ')}></div>;
};
