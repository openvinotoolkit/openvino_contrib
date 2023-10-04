interface SummarizationSectionProps {
  quoteStyle: string;
}

export function SummarizationSection({ quoteStyle }: SummarizationSectionProps): JSX.Element {
  return (
    <section className="summarization-section">
      <h3>Summarization</h3>
      <span>
        To use summarization for docstrings, start typing docstring quotes (<code>{quoteStyle}</code>).
      </span>
    </section>
  );
}
