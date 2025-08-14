"""
Sudoku Solver using CNN - Streamlit Application
"""
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys

# Set page config
st.set_page_config(page_title="Sudoku Solver", page_icon="🔢", layout="wide")

# Add custom CSS
st.markdown("""
<style>
    .sudoku-grid {
        display: grid;
        grid-template-columns: repeat(9, 1fr);
        gap: 1px;
        background: #333;
        border: 2px solid #333;
        max-width: 500px;
        margin: 0 auto;
    }
    .sudoku-cell {
        aspect-ratio: 1;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .sudoku-cell input {
        width: 100%;
        height: 100%;
        border: none;
        text-align: center;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

class SudokuSolver:
    def __init__(self):
        self.grid = np.zeros((9, 9), dtype=int)
    
    def is_valid(self, grid, row, col, num):
        # Check row and column
        if num in grid[row] or num in grid[:, col]:
            return False
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        return num not in grid[box_row:box_row+3, box_col:box_col+3]
    
    def find_empty(self, grid):
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    return (i, j)
        return None
    
    def solve(self, grid):
        empty = self.find_empty(grid)
        if not empty:
            return True
        row, col = empty
        for num in range(1, 10):
            if self.is_valid(grid, row, col, num):
                grid[row][col] = num
                if self.solve(grid):
                    return True
                grid[row][col] = 0
        return False

def create_sudoku_grid():
    grid = []
    cols = st.columns(9)
    for i in range(9):
        row = []
        for j in range(9):
            with cols[j]:
                val = st.text_input(f"cell-{i}-{j}", "", key=f"cell-{i}-{j}", 
                                  max_chars=1, help=f"Cell ({i+1}, {j+1})")
                row.append(int(val) if val.isdigit() and 1 <= int(val) <= 9 else 0)
        grid.append(row)
    return np.array(grid)

def main():
    st.title("🔢 Sudoku Solver")
    
    # Initialize solver
    if 'solver' not in st.session_state:
        st.session_state.solver = SudokuSolver()
    if 'solution' not in st.session_state:
        st.session_state.solution = None
    
    # Input section
    st.header("Enter Sudoku Puzzle")
    st.session_state.grid = create_sudoku_grid()
    
    # Solve button
    if st.button("🔍 Solve Sudoku"):
        with st.spinner("Solving..."):
            grid_copy = st.session_state.grid.copy()
            if st.session_state.solver.solve(grid_copy):
                st.session_state.solution = grid_copy
                st.success("Solution found!")
            else:
                st.error("No solution exists!")
    
    # Display solution
    if st.session_state.solution is not None:
        st.header("Solution")
        st.markdown("<div class='sudoku-grid'>", unsafe_allow_html=True)
        for i in range(9):
            for j in range(9):
                val = st.session_state.solution[i][j]
                orig_val = st.session_state.grid[i][j]
                color = "blue" if orig_val == 0 else "black"
                st.markdown(f"<div class='sudoku-cell' style='color:{color}'>{val if val != 0 else ''}</div>", 
                           unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("🔄 Reset"):
            st.session_state.solution = None
            st.rerun()

if __name__ == "__main__":
    main()
