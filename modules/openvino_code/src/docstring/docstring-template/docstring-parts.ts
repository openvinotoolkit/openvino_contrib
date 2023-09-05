export interface Decorator {
  name: string;
}

export interface Argument {
  var: string;
  type?: string;
}

export interface KeywordArgument {
  default: string;
  var: string;
  type?: string;
}

export interface Exception {
  type?: string;
}

export interface Returns {
  type?: string;
}

export interface Yields {
  type?: string;
}

export interface DocstringParts {
  name: string;
  summary?: string;
  decorators: Decorator[];
  args: Argument[];
  kwargs: KeywordArgument[];
  exceptions: Exception[];
  returns?: Returns;
  yields?: Yields;
  code: string[];
}
