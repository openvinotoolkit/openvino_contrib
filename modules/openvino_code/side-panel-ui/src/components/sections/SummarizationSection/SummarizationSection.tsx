export function SummarizationSection(): JSX.Element {
  return (
    <section className="summarization-section">
      <h3>Summarization</h3>
      <span>
        {/* TODO Consider geting selected docstring quotes from extension settings */}
        To use summarization for docstrings, start typing docstring quotes (<code>&quot;&quot;&quot;</code> by default).
      </span>
    </section>
  );
}
