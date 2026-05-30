import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, test, vi } from 'vitest';
import { Navbar } from './Navbar';
import { MemoryRouter } from 'react-router-dom';

describe('Navbar', () => {
  beforeEach(() => {
    window.open = vi.fn();
  });

  test('opens coffee URL in new tab when clicked', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <Navbar activeView="predictions" />
      </MemoryRouter>
    );

    const coffeeLink = screen.getByRole('link', { name: /coffee/i });
    await user.click(coffeeLink);

    expect(coffeeLink).toHaveAttribute(
      'href',
      'https://www.buymeacoffee.com/jakmate'
    );
    expect(coffeeLink).toHaveAttribute('target', '_blank');
    expect(coffeeLink).toHaveAttribute('rel', 'noopener noreferrer');
  });
});
