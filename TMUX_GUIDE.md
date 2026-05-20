# Tmux Quick Start Guide for EXACT 2026

`tmux` (Terminal Multiplexer) allows you to manage multiple terminal sessions in one window and keeps them running in the background even if your SSH connection drops. This is highly recommended when running long training or inference jobs on the server.

## 1. Starting and Managing Sessions

- **Start a new session**: 
  ```bash
  tmux
  ```
- **Start a new named session** (recommended for organizing tasks): 
  ```bash
  tmux new -s server_run
  ```
- **Detach from a session** (leaves it running in the background): 
  Press `Ctrl+b`, then release both and press `d`.
- **List running sessions**: 
  ```bash
  tmux ls
  ```
- **Reattach to a session**: 
  ```bash
  tmux attach -t server_run
  ```
  *(Or just `tmux a` to attach to the most recent one).*
- **Kill a session**: 
  ```bash
  tmux kill-session -t server_run
  ```

## 2. Basic Shortcuts inside Tmux

All shortcuts in tmux require a "prefix key" first. By default, this is **`Ctrl+b`**.

### Panes (Splitting the window)
- `Ctrl+b` then `%` : Split pane vertically (left/right)
- `Ctrl+b` then `"` : Split pane horizontally (top/bottom)
- `Ctrl+b` then `Arrow Keys` : Move between panes
- `Ctrl+b` then `x` : Close the current pane
- `Ctrl+b` then `z` : Toggle full-screen for the current pane (zoom in/out)

### Windows (Tabs)
- `Ctrl+b` then `c` : Create a new window (tab)
- `Ctrl+b` then `n` : Go to the next window
- `Ctrl+b` then `p` : Go to the previous window
- `Ctrl+b` then `0-9` : Jump to a specific window by number

## 3. Scrolling in Tmux

To scroll up through long output logs:
1. Press `Ctrl+b` then `[` to enter "copy mode".
2. Use the `Arrow Keys` or `Page Up` / `Page Down` to scroll.
3. Press `q` to exit copy mode and return to normal input.

## 4. Enabling Mouse Support (Optional)

If you prefer to use the mouse to scroll and select panes, you can enable mouse mode by running this command while inside a tmux session:
```bash
tmux set -g mouse on
```
*(To make this permanent, you can add `set -g mouse on` to a `~/.tmux.conf` file in your home directory).*
