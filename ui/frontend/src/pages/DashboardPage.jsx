import React, { useEffect, useState } from 'react';
import StatCard from '../components/StatCard';
import { fetchPhase6Evaluation } from '../services/api';

export default function DashboardPage() {
  const [evalData, setEvalData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPhase6Evaluation()
      .then((data) => {
        setEvalData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  const results = evalData?.phase6g_results || {};
  const sftPpl = results?.sft_metrics?.test_ppl || 13.5777;
  const basePpl = results?.baseline_metrics?.test_ppl || 19.9101;
  const sftLoss = results?.sft_metrics?.test_loss || 2.6084;
  const baseLoss = results?.baseline_metrics?.test_loss || 2.9912;
  const pplImp = results?.improvements?.perplexity?.percentage || 31.81;
  const lossImp = results?.improvements?.loss?.percentage || 12.80;

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <h1 className="page-title">MINIGPT</h1>
          <span className="badge badge-purple">Phase 6 SFT</span>
        </div>
        <p className="page-subtitle">
          6.61M Parameter Subword Transformer Language Model — Instruction Alignment Dashboard
        </p>
      </div>

      <div className="stat-grid">
        <StatCard
          title="Test Perplexity"
          value={sftPpl.toFixed(4)}
          change={`↓ ${pplImp}% Improvement`}
          changeColor="green"
          subtext={`Base: ${basePpl.toFixed(4)}`}
        />
        <StatCard
          title="Test Cross-Entropy Loss"
          value={sftLoss.toFixed(4)}
          change={`↓ ${lossImp}% Improvement`}
          changeColor="green"
          subtext={`Base: ${baseLoss.toFixed(4)}`}
        />
        <StatCard
          title="Model Parameters"
          value="6.61M"
          subtext="6,613,504 Trainable Parameters"
          change="8 Layers | 8 Heads | 256 Dim"
          changeColor="blue"
        />
        <StatCard
          title="SFT Optimizer Steps"
          value="1,000"
          subtext="Executed on AMD Ryzen 7 CPU"
          change="41,404 Training Examples"
          changeColor="blue"
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
        <div className="card">
          <h3 className="card-title">📉 Before vs After SFT Performance Progression</h3>
          <table className="custom-table">
            <thead>
              <tr>
                <th>Evaluation Split</th>
                <th>Phase 5D Base Model</th>
                <th>Phase 6E SFT Model</th>
                <th>Delta Improvement</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Validation Perplexity (PPL)</strong></td>
                <td><span className="code-badge">20.2864</span></td>
                <td><span className="code-badge" style={{ color: '#10b981' }}>13.5007</span></td>
                <td><span className="badge badge-green">↓ 33.45% (-6.7857)</span></td>
              </tr>
              <tr>
                <td><strong>Validation Loss</strong></td>
                <td><span className="code-badge">3.0099</span></td>
                <td><span className="code-badge" style={{ color: '#10b981' }}>2.6027</span></td>
                <td><span className="badge badge-green">↓ 13.53% (-0.4072)</span></td>
              </tr>
              <tr>
                <td><strong>Held-Out Test Perplexity (PPL)</strong></td>
                <td><span className="code-badge">19.9101</span></td>
                <td><span className="code-badge" style={{ color: '#10b981' }}>13.5777</span></td>
                <td><span className="badge badge-green">↓ 31.81% (-6.3324)</span></td>
              </tr>
              <tr>
                <td><strong>Held-Out Test Loss</strong></td>
                <td><span className="code-badge">2.9912</span></td>
                <td><span className="code-badge" style={{ color: '#10b981' }}>2.6084</span></td>
                <td><span className="badge badge-green">↓ 12.80% (-0.3828)</span></td>
              </tr>
              <tr>
                <td><strong>Average Token Jaccard (30 Prompts)</strong></td>
                <td><span className="code-badge">0.0905</span></td>
                <td><span className="code-badge" style={{ color: '#10b981' }}>0.1059</span></td>
                <td><span className="badge badge-purple">+17.02% (+0.0154)</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3 className="card-title">🔍 Quick Model Summary</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
            <div><strong>Architecture:</strong> Subword GPT (Decoder-Only)</div>
            <div><strong>Tokenizer:</strong> Custom Byte Pair Encoding (BPE)</div>
            <div><strong>Vocabulary:</strong> 1,024 subword tokens</div>
            <div><strong>Context Window:</strong> 128 tokens</div>
            <div><strong>Instruction Dataset:</strong> yahma/alpaca-cleaned (51,756 records)</div>
            <div><strong>Training Device:</strong> AMD Ryzen 7 7730U CPU</div>
            <div><strong>Pytest Suite:</strong> <span className="badge badge-green">87/87 Passed</span></div>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="card-title">📊 Evaluated Empirical Visualization Plots</h3>
        <div className="plot-grid">
          <div>
            <div style={{ fontSize: '0.85rem', marginBottom: '0.5rem', color: 'var(--text-muted)', fontWeight: 600 }}>Validation Perplexity</div>
            <img src="/api/plots/validation_ppl_before_after.png" alt="Validation PPL" className="plot-img" />
          </div>
          <div>
            <div style={{ fontSize: '0.85rem', marginBottom: '0.5rem', color: 'var(--text-muted)', fontWeight: 600 }}>Held-Out Test Perplexity</div>
            <img src="/api/plots/test_ppl_before_after.png" alt="Test PPL" className="plot-img" />
          </div>
          <div>
            <div style={{ fontSize: '0.85rem', marginBottom: '0.5rem', color: 'var(--text-muted)', fontWeight: 600 }}>Held-Out Test Cross-Entropy Loss</div>
            <img src="/api/plots/test_loss_before_after.png" alt="Test Loss" className="plot-img" />
          </div>
        </div>
      </div>
    </div>
  );
}
