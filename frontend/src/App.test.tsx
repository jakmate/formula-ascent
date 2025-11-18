import { describe, it, vi } from 'vitest';
import { render } from '@testing-library/react';
import App from './App';

vi.mock('./components/Dashboard', () => ({
  default: () => <div data-testid="dashboard">Dashboard Component</div>,
}));

describe('App', () => {
  it('renders without crashing', () => {
    render(<App />);
  });
});
