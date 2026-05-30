import { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Navbar } from './Navbar';
import { Schedule } from './schedule/Schedule';
import { PredictionsTable } from './table/PredictionsTable';

const Dashboard = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const activeView =
    location.pathname === '/schedule'
      ? ('schedule' as const)
      : ('predictions' as const);

  useEffect(() => {
    if (location.pathname === '/') {
      navigate('/predictions', { replace: true });
    }
  }, [location.pathname, navigate]);

  return (
    <div className="min-h-screen bg-[#0A0A0A]">
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCI+PHBhdGggZD0iTTQwIDBIMHY0MCIgc3Ryb2tlPSIjMkEyQTJBIiBzdHJva2Utd2lkdGg9IjAiLz48cGF0aCBkPSJNMCAwaDQwdjQwSDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTAgMGgxdjFIMHpNMzkgMzloMXYxaC0xeiIgZmlsbD0iIzJBMkEyQSIvPjwvc3ZnPg==')] opacity-20 pointer-events-none"></div>

      <Navbar activeView={activeView} />

      <div className="max-w-7xl mx-auto p-4 relative z-10">
        {activeView === 'predictions' && <PredictionsTable />}
        {activeView === 'schedule' && <Schedule />}
      </div>
    </div>
  );
};

export default Dashboard;
