import React, { useEffect, useState } from 'react';
import { fetchPhase6Evaluation } from '../services/api';

export default function EvaluationPage() {
  const [evalData, setEvalData] = useState(null);
  const [selectedPromptId, setSelectedPromptId] = useState(1);

  useEffect(() => {
    fetchPhase6Evaluation()
      .then((data) => setEvalData(data))
      .catch(console.error);
  }, []);

  const results = evalData?.phase6g_results || {};
  const comparison = evalData?.phase6h_comparison || {};
  const qualSamples = comparison?.qualitative_comparisons || [];

  const selectedSample = qualSamples.find((s) => s.id === selectedPromptId) || qualSamples[0] || {
    id: 1,
    category: 'factual question',
    instruction: 'What is the capital of France?',
    input: '',
    baseline_response: '- Students (1999)\n- Stephysical Students (1999)\n-',
    sft_response: 'The United States of States and American American American American American',
    observation: 'SFT model uses structured English preamble, though hallucinates country name. Baseline produces unformatted bibliography bullet points.'
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">🧪 Phase 6 Evaluation & Benchmarks</h1>
        <p className="page-subtitle">
          Independently verified validation, held-out test dataset metrics, and fixed prompt suite comparisons.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
        <div className="card">
          <h3 className="card-title">📊 Validation Set (5,176 Examples)</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px' }}>
              <span>Phase 5D Base Validation PPL</span>
              <span className="code-badge">20.2864</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px' }}>
              <span>Phase 6E SFT Validation PPL</span>
              <span className="code-badge" style={{ color: '#10b981' }}>13.5007</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '8px', color: '#6ee7b7' }}>
              <span>Validation Perplexity Improvement</span>
              <strong>↓ 33.45% (-6.7857 PPL)</strong>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="card-title">🎯 Held-Out Test Set (5,176 Examples)</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px' }}>
              <span>Phase 5D Base Test PPL</span>
              <span className="code-badge">19.9101</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px' }}>
              <span>Phase 6E SFT Test PPL</span>
              <span className="code-badge" style={{ color: '#10b981' }}>13.5777</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '8px', color: '#6ee7b7' }}>
              <span>Held-Out Test Perplexity Improvement</span>
              <strong>↓ 31.81% (-6.3324 PPL)</strong>
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: '2rem' }}>
        <h3 className="card-title">🔍 Interactive Qualitative Prompt Inspector</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
          Select a sample instruction from the 30 fixed evaluation prompts to examine the exact unedited baseline vs SFT outputs:
        </p>

        <div className="form-group" style={{ marginBottom: '1.5rem' }}>
          <label className="form-label">Select Prompt Category / Instruction</label>
          <select
            className="form-select"
            value={selectedPromptId}
            onChange={(e) => setSelectedPromptId(parseInt(e.target.value))}
          >
            {qualSamples.map((s) => (
              <option key={s.id} value={s.id}>
                [{s.category.toUpperCase()}] ID {s.id}: {s.instruction}
              </option>
            ))}
          </select>
        </div>

        <div style={{ background: 'rgba(11, 15, 25, 0.8)', padding: '1.25rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
          <div style={{ marginBottom: '1rem' }}>
            <span className="badge badge-purple" style={{ marginRight: '0.5rem' }}>{selectedSample.category}</span>
            <strong>Instruction:</strong> {selectedSample.instruction}
            {selectedSample.input && <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Input: <code>{selectedSample.input}</code></div>}
          </div>

          <div className="comparison-grid">
            <div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '0.4rem' }}>
                Baseline Model Response (Phase 5D)
              </div>
              <div className="response-box" style={{ minHeight: '100px', fontSize: '0.85rem' }}>
                {selectedSample.baseline_response}
              </div>
            </div>

            <div>
              <div style={{ fontSize: '0.85rem', color: '#6ee7b7', fontWeight: 600, marginBottom: '0.4rem' }}>
                SFT Model Response (Phase 6E)
              </div>
              <div className="response-box" style={{ minHeight: '100px', fontSize: '0.85rem' }}>
                {selectedSample.sft_response}
              </div>
            </div>
          </div>

          <div style={{ marginTop: '1rem', padding: '0.85rem', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '8px', fontSize: '0.85rem', borderLeft: '3px solid var(--accent-primary)' }}>
            💡 <strong>Factual Observation:</strong> {selectedSample.observation}
          </div>
        </div>
      </div>
    </div>
  );
}
