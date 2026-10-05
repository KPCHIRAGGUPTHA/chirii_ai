import React from 'react';

export default function TrainingPage() {
  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">⚙ Training Setup & SFT Hyperparameters</h1>
        <p className="page-subtitle">
          Supervised Fine-Tuning (Phase 6E) execution details, hardware environment, and context-length truncation audit.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
        <div className="card">
          <h3 className="card-title">🎛 Phase 6E SFT Hyperparameters</h3>
          <table className="custom-table">
            <tbody>
              <tr>
                <td><strong>Optimizer Steps</strong></td>
                <td><span className="code-badge">1,000 Steps</span></td>
              </tr>
              <tr>
                <td><strong>Learning Rate</strong></td>
                <td><span className="code-badge">1e-4</span> (Min LR: 1e-5)</td>
              </tr>
              <tr>
                <td><strong>Learning Rate Scheduler</strong></td>
                <td>Linear Warmup (20 steps) + Cosine Decay</td>
              </tr>
              <tr>
                <td><strong>Micro Batch Size</strong></td>
                <td><span className="code-badge">8 Sequences</span></td>
              </tr>
              <tr>
                <td><strong>Gradient Accumulation</strong></td>
                <td><span className="code-badge">4 Steps</span> (Effective batch = 32)</td>
              </tr>
              <tr>
                <td><strong>Weight Decay</strong></td>
                <td><span className="code-badge">0.01</span></td>
              </tr>
              <tr>
                <td><strong>Gradient Clipping</strong></td>
                <td><span className="code-badge">1.0</span></td>
              </tr>
              <tr>
                <td><strong>Label Masking</strong></td>
                <td><span className="code-badge">ignore_index=-100</span> (Response loss only)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3 className="card-title">💻 Hardware Environment & Execution Stats</h3>
          <table className="custom-table">
            <tbody>
              <tr>
                <td><strong>Hardware Host</strong></td>
                <td>AMD Ryzen 7 7730U CPU</td>
              </tr>
              <tr>
                <td><strong>Execution Threads</strong></td>
                <td>8 Threads (`torch.set_num_threads(8)`)</td>
              </tr>
              <tr>
                <td><strong>CUDA / GPU Status</strong></td>
                <td>Strict CPU Mode (`CUDA = False`)</td>
              </tr>
              <tr>
                <td><strong>Total Training Time</strong></td>
                <td><span className="code-badge">4,167.88 sec</span> (69.46 minutes)</td>
              </tr>
              <tr>
                <td><strong>Average Step Duration</strong></td>
                <td><span className="code-badge">4.11 sec/step</span></td>
              </tr>
              <tr>
                <td><strong>Active Tokens Processed</strong></td>
                <td><span className="code-badge">1,713,142</span> Response Tokens</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h3 className="card-title">📏 Phase 6B Context-Length Audit Analysis</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
          Empirical truncation study conducted across all 51,756 instruction-response sequences:
        </p>
        <table className="custom-table">
          <thead>
            <tr>
              <th>Block Size</th>
              <th>Complete Fit (%)</th>
              <th>Retained Tokens (%)</th>
              <th>Content Loss (%)</th>
              <th>Selection Decision</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ background: 'rgba(16, 185, 129, 0.05)' }}>
              <td><strong>128 Tokens</strong></td>
              <td><span className="code-badge">25.03%</span></td>
              <td><span className="code-badge">29.35%</span></td>
              <td><span className="code-badge">70.65%</span></td>
              <td><span className="badge badge-green">Selected (Matches Pretrained Positional Embeddings)</span></td>
            </tr>
            <tr>
              <td><strong>256 Tokens</strong></td>
              <td><span className="code-badge">43.64%</span></td>
              <td><span className="code-badge">49.50%</span></td>
              <td><span className="code-badge">50.50%</span></td>
              <td><span className="badge badge-purple">Exceeds Model C Positional Matrix</span></td>
            </tr>
            <tr>
              <td><strong>512 Tokens</strong></td>
              <td><span className="code-badge">67.38%</span></td>
              <td><span className="code-badge">76.93%</span></td>
              <td><span className="code-badge">23.07%</span></td>
              <td><span className="badge badge-purple">Exceeds Model C Positional Matrix</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
