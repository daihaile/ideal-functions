# 📋 Python Project Implementation Plan

This file tracks the development process for the `DLMDSPWP01` assignment.

## Phase 1: Setup & Data Loading
**Goal:** Get all provided data (A, C) loaded into your SQLite database in the correct structure.

- [ ] **Step 1.1: Project Setup**
    - [x] Create main project folder.
    - [x] Add `GUIDELINES.md`.
    - [x] Create `data/` folder (for CSVs).
    - [x] Create main Python script (`main.py`) or package structure.
    - [x] Set up Python environment (e.g., Pipenv) and install `pandas`, `sqlalchemy`, `bokeh`.

- [ ] **Step 1.2: Plan OOP Structure**
    - [x] **Plan Classes:**
        - `DatabaseManager`: Handles SQLAlchemy connection, setup, table creation.
        - `DataLoader`: Base class (for inheritance requirement).
        - `TrainingLoader` (inherits from `DataLoader`): Loads training data (A).
        - `IdealFunctionLoader` (inherits from `DataLoader`): Loads ideal functions (C).
        - `TestProcessor`: Loads and processes test data (B).

- [X] **Step 1.3: Implement `DatabaseManager`**
    - [X] Use SQLAlchemy to define schemas for **Table 1**, **Table 2**, and **Table 3**.
    - [X] Create a method to connect to the SQLite file (e.g., `db.sqlite`) and create tables.

- [X] **Step 1.4: Implement `TrainingLoader`**
    - [X] Read the 4 training CSVs (A) using `pandas`.
    - [X] Combine into a single DataFrame matching **Table 1** (X, Y1, Y2, Y3, Y4).
    - [X] Use `DatabaseManager` to write this DataFrame to Table 1.

- [ ] **Step 1.5: Implement `IdealFunctionLoader`**
    - [X] Read the ideal functions CSV (C) using `pandas`.
    - [X] Ensure data format matches **Table 2** (X, Y1... Y50).
    - [X] Use `DatabaseManager` to write this DataFrame to Table 2.

---

## Phase 2: Core Analysis (Finding Best-Fit Functions)
**Goal:** Find the 4 best ideal functions and the "max deviation" thresholds.

- [ ] **Step 2.1: Create `Analyzer` Class**
    - [ ] Create a new class for analysis logic, reading from Table 1 and Table 2.

- [ ] **Step 2.2: Implement Least-Squares Method**
    - [ ] Write a method to calculate the **sum of squared deviations**.
    - [ ] **Loop 1:** For each of the 4 training functions (Y1-Y4 in Table 1).
    - [ ] **Loop 2:** Compare it against *all 50* ideal functions (Y1-Y50 in Table 2).
    - [ ] Store the sum-of-squares result for each comparison.

- [ ] **Step 2.3: Select 4 Best Functions**
    - [ ] For each training function, find the ideal function with the **minimum** sum of squares.
    - [ ] Store these 4 chosen ideal functions (e.g., as a dictionary mapping).

- [ ] **Step 2.4: Calculate Max Deviation Thresholds**
    - [ ] For each of the 4 "best-fit" pairs:
    - [ ] Calculate `abs(y_train - y_ideal)` for every x-value.
    - [ ] Find the **largest deviation** (`max`) in that set.
    - [ ] Store these 4 "max training deviation" values.

---

## Phase 3: Test Data Mapping
**Goal:** Process the test data (B) and save mapped results to Table 3.

- [ ] **Step 3.1: Implement `TestProcessor`**
    - [ ] Create a method to load the test data (B) "line-by-line" from its CSV.

- [ ] **Step 3.2: Implement Mapping Logic**
    - [ ] For each x-y pair from the test data:
    - [ ] Compare it against your **4 chosen ideal functions**.
    - [ ] **Apply Criterion:** Check if `abs(y_test - y_chosen_ideal) <= (max_training_deviation * sqrt(2))`.
    - [ ] If it maps, choose the ideal function with the *smallest* deviation.

- [ ] **Step 3.3: Save Mapped Results**
    - [ ] If a test point is mapped, save its data to **Table 3**.
    - [ ] Row must contain: `X`, `Y`, `Delta Y` (the deviation), and `No. of ideal func`.

---

## Phase 4: Visualization, Testing & Documentation
**Goal:** Fulfill all remaining code quality, visualization, and documentation requirements.

- [ ] **Step 4.1: Implement `Visualizer` Class (Bokeh)**
    - [ ] Create a class that uses **Bokeh** to generate plots.
    - [ ] **Plot 1:** Plot 4 training functions, each overlaid with its "best-fit" ideal function.
    - [ ] **Plot 2:** Plot test data (scatter plot). Use different colors for mapped vs. unmapped points.
    - [ ] **Plot 3:** Create a plot of deviations (e.g., histogram of `Delta Y` from Table 3).

- [ ] **Step 4.2: Write Unit Tests**
    - [ ] Create a `tests/` folder with `pytest`.
    - [ ] Test the Least-Squares calculation.
    - [ ] Test the `sqrt(2)` mapping logic.
    - [ ] Test database table creation.

- [ ] **Step 4.3: Implement Exception Handling**
    - [ ] Add `try...except FileNotFoundError` for file I/O.
    - [ ] Add `try...except` for database errors (e.g., `sqlalchemy.exc.OperationalError`).
    - [ ] Create one **user-defined exception** (e.g., `DataMismatchError`) and raise it.

- [ ] **Step 4.4: Write Documentation (Continuous)**
    - [ ] Add **docstrings** to *all* classes and methods.
    - [ ] Add inline comments for complex logic.

---

## Phase 5: Final Review & Git Task
**Goal:** Finalize the project and prepare the written submission components.

- [ ] **Step 5.1: Review Guidelines**
    - [ ] Go through `GUIDELINES.md` and check every box.

- [ ] **Step 5.2: Write Answer for Git Task**
    - [ ] In the written assignment, answer the "Additional Task" with the required Git commands.

- [ ] **Step 5.3: Prepare Code for Appendix**
    - [ ] Copy/paste the entire, well-documented source code into the assignment's appendix.