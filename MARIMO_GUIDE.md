# Complete Guide to Marimo

A comprehensive guide to understanding and using Marimo - a reactive Python notebook framework.

## Table of Contents

1. [What is Marimo?](#what-is-marimo)
2. [Key Features and Capabilities](#key-features-and-capabilities)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
5. [Reactive Programming Model](#reactive-programming-model)
6. [Advanced Features](#advanced-features)
7. [Integration with Streamlit](#integration-with-streamlit)
8. [Best Practices](#best-practices)
9. [Examples](#examples)
10. [Comparison with Jupyter](#comparison-with-jupyter)
11. [Troubleshooting](#troubleshooting)

---

## What is Marimo?

**Marimo** is a reactive Python notebook framework that combines the interactivity of Jupyter notebooks with the power of reactive programming. Unlike traditional notebooks, Marimo automatically tracks dependencies between cells and re-executes them when their inputs change.

### Key Concepts

- **Reactive**: Cells automatically update when their dependencies change
- **No Hidden State**: All state is explicit and tracked
- **Type-Safe**: Better error detection and type checking
- **Reproducible**: Deterministic execution order
- **Shareable**: Can be exported to HTML or Python scripts

---

## Key Features and Capabilities

### 1. **Reactive Execution**

Marimo automatically tracks dependencies between cells and re-executes them when inputs change:

```python
# Cell 1
import marimo

__all__ = ["x", "y"]

# Cell 2
x = 10

# Cell 3 (automatically re-runs when x changes)
y = x * 2  # y = 20

# If you change x to 15, y automatically becomes 30
```

### 2. **No Hidden State**

Unlike Jupyter, Marimo makes all state explicit. You can't accidentally use variables from previous runs.

### 3. **Type Checking**

Marimo provides better type checking and error detection:

```python
# Marimo will catch type errors early
def add(a: int, b: int) -> int:
    return a + b

result = add("hello", "world")  # Type error caught!
```

### 4. **UI Components**

Marimo includes built-in UI components for interactive notebooks:

```python
import marimo.ui as mui

# Slider
slider = mui.slider(0, 100, value=50)

# Text input
text_input = mui.text_input("Enter your name")

# Button
button = mui.button("Click me")

# Dropdown
dropdown = mui.dropdown(["Option 1", "Option 2", "Option 3"])
```

### 5. **Data Visualization**

Built-in support for plotting and visualization:

```python
import marimo
import pandas as pd
import numpy as np

# Create data
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})

# Plot (Marimo automatically displays plots)
marimo.plot(data, kind='scatter', x='x', y='y')
```

### 6. **Export Capabilities**

- Export to HTML (standalone, interactive)
- Export to Python script
- Share via URL (with Marimo Cloud)

### 7. **Version Control Friendly**

Marimo notebooks are stored as Python files, making them:
- Easy to version control
- Merge-friendly
- Reviewable in pull requests

---

## Installation

### Using UV (Recommended)

```bash
uv add marimo
```

### Using pip

```bash
pip install marimo
```

### Using conda

```bash
conda install -c conda-forge marimo
```

### Verify Installation

```bash
marimo --version
```

---

## Basic Usage

### Starting a Marimo Notebook

```bash
# Create a new notebook
marimo edit my_notebook.py

# Or open existing notebook
marimo edit notebook.py
```

### Basic Cell Structure

```python
# Marimo notebooks are Python files with special markers
import marimo

__all__ = ["variable1", "variable2"]

# Cell 1: Import libraries
import numpy as np
import pandas as pd

# Cell 2: Define variables
variable1 = 10
variable2 = "Hello, Marimo!"

# Cell 3: Use variables (automatically reactive)
result = variable1 * 2
print(f"{variable2}: {result}")
```

### Running a Notebook

```bash
# Run notebook in edit mode
marimo edit notebook.py

# Run notebook as app
marimo run notebook.py

# Serve notebook as web app
marimo serve notebook.py
```

---

## Reactive Programming Model

### How Reactivity Works

Marimo tracks dependencies using variable names:

```python
# Cell 1
import marimo

__all__ = ["x", "y", "z"]

# Cell 2
x = 5

# Cell 3 (depends on x)
y = x * 2  # y = 10

# Cell 4 (depends on y)
z = y + 1  # z = 11

# If you change x to 10:
# - Cell 2 re-runs: x = 10
# - Cell 3 automatically re-runs: y = 20
# - Cell 4 automatically re-runs: z = 21
```

### Explicit Dependencies

You must explicitly declare what variables a cell uses:

```python
# Cell 1
import marimo

__all__ = ["data", "processed_data"]

# Cell 2
data = [1, 2, 3, 4, 5]

# Cell 3
processed_data = [x * 2 for x in data]  # Explicitly uses 'data'
```

### Preventing Infinite Loops

Marimo prevents circular dependencies:

```python
# This will cause an error
x = y + 1  # Error: y not defined yet
y = x + 1  # Error: circular dependency
```

---

## Advanced Features

### 1. **UI Components**

Marimo provides rich UI components:

```python
import marimo.ui as mui

# Slider with callback
def on_slider_change(value):
    print(f"Slider value: {value}")

slider = mui.slider(
    min=0,
    max=100,
    value=50,
    on_change=on_slider_change
)

# Multiple inputs
form = mui.form({
    "name": mui.text_input("Name"),
    "age": mui.number_input("Age", min=0, max=120),
    "email": mui.text_input("Email", type="email")
})
```

### 2. **Data Tables**

Interactive data tables:

```python
import marimo
import pandas as pd

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [85, 92, 78]
})

# Display interactive table
marimo.table(df)
```

### 3. **Plotting**

Multiple plotting options:

```python
import marimo
import matplotlib.pyplot as plt
import numpy as np

# Using matplotlib
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Sine Wave")
marimo.display(plt)  # Display in notebook

# Using plotly (if installed)
import plotly.express as px
fig = px.scatter(df, x='Age', y='Score')
marimo.display(fig)
```

### 4. **Async Operations**

Support for async/await:

```python
import asyncio
import marimo

async def fetch_data():
    await asyncio.sleep(1)
    return {"data": [1, 2, 3, 4, 5]}

# In a cell
result = await fetch_data()
print(result)
```

### 5. **State Management**

Managing state across cells:

```python
# Cell 1: Initialize state
import marimo

__all__ = ["state"]

class AppState:
    def __init__(self):
        self.counter = 0
        self.data = []

state = AppState()

# Cell 2: Update state
state.counter += 1
state.data.append(state.counter)

# Cell 3: Display state
print(f"Counter: {state.counter}")
print(f"Data: {state.data}")
```

### 6. **Custom Components**

Create custom UI components:

```python
import marimo.ui as mui

class CustomCounter(mui.Component):
    def __init__(self, initial_value=0):
        self.value = initial_value
        self.button = mui.button("Increment", on_click=self.increment)
    
    def increment(self):
        self.value += 1
        return self.value
    
    def render(self):
        return mui.vstack([
            mui.text(f"Count: {self.value}"),
            self.button
        ])

counter = CustomCounter()
```

---

## Integration with Streamlit

### Basic Integration

Marimo can be integrated into Streamlit apps:

```python
import streamlit as st
import marimo

# Initialize Marimo
marimo.init()

# Create a Marimo cell
@marimo.cell
def marimo_computation():
    import numpy as np
    data = np.random.randn(100)
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'data': data.tolist()
    }

# Use in Streamlit
st.title("Marimo + Streamlit")
result = marimo_computation()
st.write(f"Mean: {result['mean']:.2f}")
st.write(f"Std: {result['std']:.2f}")
st.line_chart(result['data'])
```

### Advanced Integration

```python
import streamlit as st
import marimo
from marimo import App

# Create Marimo app
app = App()

@app.cell
def __():
    import pandas as pd
    import numpy as np
    return pd, np

@app.cell
def __(pd, np):
    # Generate data
    data = pd.DataFrame({
        'x': np.random.randn(1000),
        'y': np.random.randn(1000)
    })
    return data

@app.cell
def __(data):
    # Compute statistics
    stats = {
        'mean_x': float(data['x'].mean()),
        'mean_y': float(data['y'].mean()),
        'correlation': float(data['x'].corr(data['y']))
    }
    return stats

# In Streamlit
st.title("Marimo App in Streamlit")
stats = app.get_cell_output('stats')
st.metric("Mean X", f"{stats['mean_x']:.2f}")
st.metric("Mean Y", f"{stats['mean_y']:.2f}")
st.metric("Correlation", f"{stats['correlation']:.2f}")
```

---

## Best Practices

### 1. **Explicit Dependencies**

Always declare dependencies explicitly:

```python
# Good
import marimo
__all__ = ["x", "y"]
x = 10
y = x * 2

# Bad (implicit dependencies)
x = 10
y = x * 2  # Marimo might not track this correctly
```

### 2. **Organize Cells Logically**

Group related cells together:

```python
# Group 1: Imports
import marimo
__all__ = ["data", "result"]

# Group 2: Data loading
import pandas as pd
data = pd.read_csv("data.csv")

# Group 3: Processing
result = data.groupby('category').sum()
```

### 3. **Use Type Hints**

Type hints help Marimo catch errors:

```python
def process_data(data: list[int]) -> dict[str, float]:
    return {
        'sum': sum(data),
        'mean': sum(data) / len(data)
    }
```

### 4. **Avoid Side Effects**

Minimize side effects in cells:

```python
# Good
def process(x: int) -> int:
    return x * 2

result = process(10)

# Bad (side effect)
file = open("data.txt", "w")
file.write("data")  # Side effect!
```

### 5. **Use UI Components for Interactivity**

Use Marimo's UI components instead of manual input:

```python
# Good
import marimo.ui as mui
value = mui.slider(0, 100, value=50)

# Less ideal
value = int(input("Enter value: "))
```

---

## Examples

### Example 1: Data Analysis Notebook

```python
import marimo

__all__ = ["df", "summary", "plot"]

# Cell 1: Load data
import pandas as pd
df = pd.read_csv("sales_data.csv")

# Cell 2: Summary statistics
summary = df.describe()

# Cell 3: Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['sales'])
plt.title("Sales Over Time")
plot = plt
```

### Example 2: Interactive Dashboard

```python
import marimo
import marimo.ui as mui

__all__ = ["selected_year", "filtered_data", "chart"]

# Cell 1: UI Controls
selected_year = mui.slider(
    min=2020,
    max=2024,
    value=2023,
    label="Select Year"
)

# Cell 2: Filter data
import pandas as pd
df = pd.read_csv("data.csv")
filtered_data = df[df['year'] == selected_year]

# Cell 3: Display chart
import plotly.express as px
chart = px.bar(
    filtered_data,
    x='month',
    y='sales',
    title=f"Sales for {selected_year}"
)
```

### Example 3: Machine Learning Pipeline

```python
import marimo

__all__ = ["model", "predictions", "accuracy"]

# Cell 1: Load and prepare data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("iris.csv")
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Cell 2: Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Cell 3: Evaluate
predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
```

---

## Comparison with Jupyter

| Feature | Jupyter | Marimo |
|---------|---------|--------|
| **Execution Model** | Imperative | Reactive |
| **State Management** | Hidden state | Explicit state |
| **Dependency Tracking** | Manual | Automatic |
| **Reproducibility** | Difficult | Easy |
| **Version Control** | JSON files | Python files |
| **Type Checking** | Limited | Built-in |
| **UI Components** | Requires extensions | Built-in |
| **Sharing** | Requires nbconvert | Built-in export |

### When to Use Marimo

- ✅ Building interactive dashboards
- ✅ Data analysis with reactive updates
- ✅ Educational content
- ✅ Prototyping with UI components
- ✅ When reproducibility is important

### When to Use Jupyter

- ✅ Exploratory data analysis
- ✅ Quick prototyping
- ✅ When you need extensive ecosystem
- ✅ When working with existing Jupyter workflows

---

## Troubleshooting

### Common Issues

#### 1. **Cells Not Updating**

**Problem**: Cells don't automatically update when dependencies change.

**Solution**: Ensure you've declared variables in `__all__`:

```python
import marimo
__all__ = ["x", "y"]  # Declare all exported variables
```

#### 2. **Circular Dependencies**

**Problem**: Error about circular dependencies.

**Solution**: Restructure your code to avoid circular references:

```python
# Bad
x = y + 1
y = x + 1

# Good
x = 10
y = x + 1
z = y + 1
```

#### 3. **Import Errors**

**Problem**: Modules not found.

**Solution**: Install dependencies and ensure they're in the same environment:

```bash
uv add package-name
# or
pip install package-name
```

#### 4. **UI Components Not Rendering**

**Problem**: UI components don't appear.

**Solution**: Ensure you're using Marimo's UI module correctly:

```python
import marimo.ui as mui
component = mui.slider(0, 100)
# Component will render automatically in Marimo
```

### Getting Help

- **Documentation**: [marimo.io/docs](https://marimo.io/docs)
- **GitHub**: [github.com/marimo-team/marimo](https://github.com/marimo-team/marimo)
- **Discord**: Marimo community Discord
- **Issues**: GitHub Issues

---

## Resources

### Official Resources

- **Website**: [marimo.io](https://marimo.io)
- **Documentation**: [marimo.io/docs](https://marimo.io/docs)
- **GitHub**: [github.com/marimo-team/marimo](https://github.com/marimo-team/marimo)
- **Examples**: [marimo.io/examples](https://marimo.io/examples)

### Learning Resources

- **Tutorials**: Check the official documentation
- **Video Tutorials**: YouTube search "Marimo Python"
- **Blog Posts**: Marimo blog for updates and tips

### Community

- **Discord**: Join the Marimo Discord community
- **GitHub Discussions**: Ask questions and share ideas
- **Twitter**: Follow @marimo_io for updates

---

## Conclusion

Marimo is a powerful reactive notebook framework that brings modern programming practices to data science and interactive computing. Its reactive model, explicit state management, and built-in UI components make it an excellent choice for building interactive applications and reproducible analyses.

### Key Takeaways

1. **Reactive**: Cells automatically update when dependencies change
2. **Explicit**: All state is explicit and tracked
3. **Reproducible**: Deterministic execution order
4. **Interactive**: Built-in UI components for interactivity
5. **Shareable**: Easy to export and share

### Next Steps

1. Install Marimo: `uv add marimo` or `pip install marimo`
2. Create your first notebook: `marimo edit my_notebook.py`
3. Explore the examples in this guide
4. Check out the official documentation
5. Join the community for support

---

**Happy coding with Marimo! 🎉**

