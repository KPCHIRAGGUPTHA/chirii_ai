import React from 'react';

export default function StatCard({ title, value, subtext, change, changeColor = 'green' }) {
  return (
    <div className="card stat-card">
      <div className="stat-label">{title}</div>
      <div className="stat-value">{value}</div>
      {change && (
        <div className={`stat-change ${changeColor}`}>
          {change}
        </div>
      )}
      {subtext && <div style={{ fontSize: '0.8rem', color: 'var(--text-subtle)' }}>{subtext}</div>}
    </div>
  );
}
