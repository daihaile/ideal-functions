# Project Guidelines: DLMDSPWP01

This file outlines the core requirements for the programming assignment[cite: 3], based on the task description. Use this as a checklist to ensure all criteria are met.

## 1. 📈 Core Analysis & Logic

- [ ] **Find Best Functions**: The program must identify the 4 ideal functions (from 50) that best fit the 4 training datasets[cite: 24].
    - **Criterion**: The selection must be based on minimizing the "sum of all y-deviations squared (Least-Square)"[cite: 32].
- [ ] **Map Test Data**: The program must process the test data (B) line-by-line[cite: 47].
    - **Criterion**: A test point is mapped if its deviation from one of the 4 chosen ideal functions does **not** exceed the "largest deviation between training dataset (A) and the ideal function (C) chosen for it by more than factor $\sqrt{2}$"[cite: 33].
- [ ] **Save Results**: If a test point is mapped, the program must save the x-y pair, the calculated deviation, and the ID of the ideal function it mapped to[cite: 27, 48].

## 2. 🗃️ Database & Data Structure

- [ ] **Database System**: Must use **SQLite** (as a file)[cite: 42].
- [ ] **ORM**: Must use **SQLAlchemy** to interact with the database[cite: 42].
- [ ] **Table 1 (Training)**: Load the 4 training datasets into a single table[cite: 42].
    - **Structure**: 5 columns: `X`, `Y1 (training func)`, `Y2(training func)`, `Y3(training func)`, `Y4(training func)`[cite: 43, 59].
- [ ] **Table 2 (Ideal)**: Load the 50 ideal functions into a separate table[cite: 44].
    - **Structure**: 51 columns: `X`, `Y1 (ideal func)`, `Y2 (ideal func)`, ..., `Y50 (ideal func)`[cite: 45, 63].
- [ ] **Table 3 (Test Results)**: Save the mapped test data results into a third table[cite: 48].
    - **Structure**: 4 columns: `X (test func)`, `Y (test func)`, `Delta Y (test func)`, `No. of ideal func`[cite: 49, 65].

## 3. 🛠️ Code Quality & Technical Requirements

- [ ] **Core Libraries**:
    - [ ] `pandas` must be used[cite: 55].
    - [ ] `sqlalchemy` must be used[cite: 55].
    - [ ] `bokeh` must be used for visualization[cite: 55].
- [ ] **Design**: Must have a "sensibly object-oriented" design[cite: 52].
- [ ] **Inheritance**: Must include "at least one inheritance"[cite: 53].
- [ ] **Error Handling**: Must include "standard- und user-defined exception handlings"[cite: 54].
- [ ] **Testing**: Must include **unit-tests** for "all useful elements"[cite: 31, 56].
- [ ] **Documentation**:
    - [ ] The entire code must be documented[cite: 57].
    - [ ] Must include **docstrings** ("Documentation Strings")[cite: 57].

## 4. 📊 Visualization

- [ ] **Tool**: Must use **Bokeh** for all data visualization[cite: 55].
- [ ] **Content**: The visualization must "logically visualize" all data[cite: 29], including:
    - [ ] The training data[cite: 50].
    - [ ] The test data[cite: 50].
    - [ ] The 4 chosen ideal functions[cite: 50].
    - [ ] The assigned/mapped test datasets[cite: 50].
    - [ ] An "appropriately chosen representation of the deviation"[cite: 50].

## 5. 📝 Additional Written Task (Git)

This is a separate written task for the paper, not part of the Python code.

- [ ] **Task**: Write out the Git commands for the following scenario:
    - [ ] Clone the `develop` branch from a remote repository[cite: 69, 70].
    - [ ] After adding a new function, introduce this change to the team's `develop` branch[cite: 71, 72].
    - [ ] Must include `commit` and `push` commands[cite: 73].
    - [ ] Must note the subsequent process (Pull Request, review, merge)[cite: 73].

## 6. 📦 Submission

- [ ] The "entire source code is expected to be contained in the appendix" of the final written assignment[cite: 78].