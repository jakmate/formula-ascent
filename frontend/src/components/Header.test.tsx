import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Header } from './Header';

describe('Header', () => {
  it('renders title', () => {
    render(<Header title="Test Title" />);

    expect(screen.getByText('Test Title')).toBeInTheDocument();
  });

  it('renders description when provided', () => {
    render(<Header title="Title" description="Test description" />);

    expect(screen.getByText('Test description')).toBeInTheDocument();
  });

  it('does not render description when not provided', () => {
    const { container } = render(<Header title="Title" />);

    expect(container.querySelector('.text-cyan-300')).not.toBeInTheDocument();
  });

  it('renders leftContent when provided', () => {
    render(
      <Header
        title="Title"
        leftContent={<div data-testid="left">Left Content</div>}
      />
    );

    expect(screen.getByTestId('left')).toBeInTheDocument();
  });

  it('renders rightContent when provided', () => {
    render(
      <Header
        title="Title"
        rightContent={<div data-testid="right">Right Content</div>}
      />
    );

    expect(screen.getByTestId('right')).toBeInTheDocument();
  });

  it('does not render rightContent container when not provided', () => {
    const { container } = render(<Header title="Title" />);

    const rightContentDiv = container.querySelector(
      '.flex.flex-col.sm\\:flex-row'
    );
    expect(rightContentDiv).not.toBeInTheDocument();
  });

  it('renders bottomContent when provided', () => {
    render(
      <Header
        title="Title"
        bottomContent={<div data-testid="bottom">Bottom Content</div>}
      />
    );

    expect(screen.getByTestId('bottom')).toBeInTheDocument();
  });

  it('does not render bottomContent container when not provided', () => {
    const { container } = render(<Header title="Title" />);

    const bottomContentDiv = container.querySelector('.mt-4');
    expect(bottomContentDiv).not.toBeInTheDocument();
  });

  it('renders all optional props together', () => {
    render(
      <Header
        title="Complete Header"
        description="Full description"
        leftContent={<div data-testid="left">Left</div>}
        rightContent={<div data-testid="right">Right</div>}
        bottomContent={<div data-testid="bottom">Bottom</div>}
      />
    );

    expect(screen.getByText('Complete Header')).toBeInTheDocument();
    expect(screen.getByText('Full description')).toBeInTheDocument();
    expect(screen.getByTestId('left')).toBeInTheDocument();
    expect(screen.getByTestId('right')).toBeInTheDocument();
    expect(screen.getByTestId('bottom')).toBeInTheDocument();
  });

  it('applies correct container styles', () => {
    const { container } = render(<Header title="Title" />);
    const headerDiv = container.firstChild as HTMLElement;

    expect(headerDiv).toHaveClass(
      'bg-gray-800/40',
      'backdrop-blur-lg',
      'rounded-xl',
      'p-6',
      'mb-6',
      'border',
      'border-cyan-500/30',
      'shadow-lg',
      'shadow-cyan-500/10'
    );
  });
});
