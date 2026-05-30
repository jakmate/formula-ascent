import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ProbabilityBar } from './ProbabilityBar';

describe('ProbabilityBar', () => {
  it('renders percentage text correctly', () => {
    const { getByText } = render(<ProbabilityBar percentage={75.5} />);
    expect(getByText('75.5%')).toBeInTheDocument();
  });

  it('sets correct width style', () => {
    const { container } = render(<ProbabilityBar percentage={45} />);
    const bar = container.querySelector('.h-full');
    expect(bar).toHaveStyle({ width: '45%' });
  });

  it('caps width at 100% for values over 100', () => {
    const { container } = render(<ProbabilityBar percentage={150} />);
    const bar = container.querySelector('.h-full');
    expect(bar).toHaveStyle({ width: '100%' });
  });
});
