import { ReactNode } from 'react';
import './Checkbox.css';

interface CheckboxProps {
  checked?: boolean;
  children: ReactNode;
  onChange: (isChecked: boolean) => void;
}

export const Checkbox = ({ checked, children, onChange }: CheckboxProps): JSX.Element => {
  const classNames = ['vscode-checkbox', 'codicon'];
  if (checked) {
    classNames.push('codicon-check');
  }
  return (
    <div className="checkbox">
      <div className={classNames.join(' ')} onClick={() => onChange(!checked)}></div>
      <span className="checkbox-label">{children}</span>
    </div>
  );
};
