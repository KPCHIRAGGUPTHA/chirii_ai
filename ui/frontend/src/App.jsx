import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import DashboardPage from './pages/DashboardPage';
import ChatPage from './pages/ChatPage';
import ComparePage from './pages/ComparePage';
import EvaluationPage from './pages/EvaluationPage';
import ModelPage from './pages/ModelPage';
import TrainingPage from './pages/TrainingPage';
import { fetchHealth } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    fetchHealth()
      .then((data) => {
        if (data.status === 'ok') setIsConnected(true);
      })
      .catch(() => setIsConnected(false));
  }, []);

  const renderContent = () => {
    switch (activeTab) {
      case 'chat':
        return <ChatPage />;
      case 'compare':
        return <ComparePage />;
      case 'dashboard':
        return <DashboardPage />;
      case 'evaluation':
        return <EvaluationPage />;
      case 'model':
        return <ModelPage />;
      case 'training':
        return <TrainingPage />;
      default:
        return <ChatPage />;
    }
  };

  return (
    <div className="app-container">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} isConnected={isConnected} />
      <main className="main-content">
        {renderContent()}
      </main>
    </div>
  );
}
