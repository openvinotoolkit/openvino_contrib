import './Select.css';

export interface SelectOptionProps<V extends string = string, L extends string = string> {
  value: V;
  label?: L;
}

const SelectOption = ({ value, label }: SelectOptionProps): JSX.Element => (
  <option value={value}>{label || value}</option>
);

interface SelectProps<V extends string, L extends string> {
  label: string;
  options: SelectOptionProps<V, L>[];
  selectedValue: V;
  disabled?: boolean;
  onChange?: (value: V) => void;
}

export const Select = <V extends string, L extends string>({
  label,
  options,
  disabled,
  onChange,
  selectedValue,
}: SelectProps<V, L>): JSX.Element => {
  return (
    <div className="select-container" aria-disabled={disabled}>
      <label>{label}:</label>
      <select
        className="select"
        onChange={(event) => onChange?.(event.target.value as V)}
        disabled={disabled}
        value={selectedValue}
      >
        {options.map(({ value, label: optionLabel }) => (
          <SelectOption key={value} value={value} label={optionLabel}></SelectOption>
        ))}
      </select>
    </div>
  );
};
