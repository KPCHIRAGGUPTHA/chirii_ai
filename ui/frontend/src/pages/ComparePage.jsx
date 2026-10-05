import React, { useState } from 'react';
import { compareModels } from '../services/api';

export default function ComparePage() {
  const [prompt, setPrompt] = useState('What is the capital of France?');
  const [maxTokens, setMaxTokens] = useState(40);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);

  const [loading, setLoading] = useState(false);
  const [comparison, setComparison] = useState(null);
  const [error, setError] = useState(null);

  const handleCompare = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await compareModels({
        prompt,
        max_new_tokens: maxTokens,
        temperature,
        top_k: topK
      });
      setComparison(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">⚖ Side-by-Side Model Comparison</h1>
        <p className="page-subtitle">
          Compare outputs from Phase 5D Pretrained Base Model vs Phase 6E SFT Fine-Tuned Model under identical prompts and sampling parameters.
        </p>
      </div>

      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="form-group">
          <label className="form-label">Shared Test Instruction / Prompt</label>
          <textarea
            className="form-textarea"
            placeholder="Type a prompt to test both models..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
          />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
          <div>
            <label className="form-label">Temperature: {temperature}</label>
            <input
              type="range"
              min="0.0"
              max="1.5"
              step="0.05"
              className="form-slider"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
            />
          </div>

          <div>
            <label className="form-label">Max Tokens: {maxTokens}</label>
            <input
              type="range"
              min="10"
              max="128"
              step="5"
              className="form-slider"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
            />
          </div>

          <div>
            <label className="form-label">Top-K: {topK}</label>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              className="form-slider"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
            />
          </div>
        </div>

        <button
          className="btn btn-primary"
          style={{ width: '100%' }}
          onClick={handleCompare}
          disabled={loading || !prompt.trim()}
        >
          {loading ? '⏳ Running Dual CPU Model Inference...' : '⚔ Execute Side-by-Side Comparison'}
        </button>

        {error && (
          <div style={{ marginTop: '1rem', color: 'var(--accent-danger)', fontSize: '0.9rem' }}>
            ⚠️ Error: {error}
          </div>
        )}
      </div>

      {comparison && (
        <div className="comparison-grid">
          {/* Base Model Card */}
          <div className="card" style={{ borderColor: 'rgba(76, 114, 176, 0.4)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 className="card-title" style={{ margin: 0 }}>Phase 5D Base Model</h3>
              <span className="badge badge-purple">Pretrained Baseline</span>
            </div>

            <div className="response-box" style={{ minHeight: '160px' }}>
              {comparison.base_response.response || <span style={{ color: 'var(--text-subtle)' }}>(Empty output)</span>}
            </div>

            <div className="response-meta">
              <div className="meta-item">Latency: <span className="meta-val">{comparison.base_response.generation_time_sec}s</span></div>
              <div className="meta-item">Tokens: <span className="meta-val">{comparison.base_response.tokens_generated}</span></div>
              <div className="meta-item">Speed: <span className="meta-val">{comparison.base_response.tokens_per_second} tok/s</span></div>
            </div>
          </div>

          {/* SFT Model Card */}
          <div className="card" style={{ borderColor: 'rgba(16, 185, 129, 0.4)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 className="card-title" style={{ margin: 0 }}>Phase 6E SFT Model</h3>
              <span className="badge badge-green">Instruction Aligned</span>
            </div>

            <div className="response-box" style={{ minHeight: '160px' }}>
              {comparison.sft_response.response || <span style={{ color: 'var(--text-subtle)' }}>(Empty output)</span>}
            </div>

            <div className="response-meta">
              <div className="meta-item">Latency: <span className="meta-val">{comparison.sft_response.generation_time_sec}s</span></div>
              <div className="meta-item">Tokens: <span className="meta-val">{comparison.sft_response.tokens_generated}</span></div>
              <div className="meta-item">Speed: <span className="meta-val">{comparison.sft_response.tokens_per_second} tok/s</span></div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
