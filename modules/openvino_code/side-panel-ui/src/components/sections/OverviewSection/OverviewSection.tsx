import './OverviewSection.css';

export function OverviewSection(): JSX.Element {
  return (
    <section className="overview-section">
      <span>
        OpenVINO Code provides the following features:
        <ul>
          <li>Inline Code Completion</li>
          <li>Summarization via docstring</li>  
          <li>Fill in the Middle Mode</li>
        </ul>
        To use OpenVINO Code please start the server.
      </span>
    </section>
  );
}
