let isGeneralTabActiveInternal: boolean = false;

export function setIsGeneralTabActive(value: boolean): void {
  isGeneralTabActiveInternal = value;
}

export function getIsGeneralTabActive(): boolean {
  return isGeneralTabActiveInternal;
}