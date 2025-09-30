# fun_thingymabob - `grayscale_lib.py` Library Documentation

## Overview

The `grayscale_lib.py` file contains two primary classes for mathematical computations:
- **Matrix:** A comprehensive matrix manipulation class supporting arithmetic, algebraic, and property-checking operations.
- **Fractions:** A robust class for rational number (fraction) arithmetic, supporting advanced manipulation and conversion.

This README provides a detailed guide to all features, methods, and usage examples for both classes.

---

## Matrix Class

A mathematical matrix class supporting:
- Creation (from lists, zeroes, identity, diagonal, random)
- Display (pretty Unicode formatting)
- Arithmetic (addition, subtraction, scalar/matrix multiplication, power)
- Properties (square, symmetric, orthogonal)
- Operations (transpose, trace, norm, elementwise exponentiation, duplication)

### Construction

```python
m = Matrix([[1, 2], [3, 4]])
```
- Accepts a list of lists, each inner list representing a row.
- All rows **must** have the same length.

### Display

```python
print(m)
```
- Outputs the matrix with Unicode brackets, aligned columns.

### Arithmetic

#### Addition

```python
m2 = Matrix([[5, 6], [7, 8]])
result = m + m2         # Matrix + Matrix
result = m + 10         # Matrix + scalar
result = 10 + m         # scalar + Matrix
```

#### Subtraction

```python
result = m - m2         # Matrix - Matrix
result = m - 2          # Matrix - scalar
result = 3 - m          # scalar - Matrix
```

#### Scalar Multiplication

```python
result = m * 2          # Matrix * scalar
result = 2 * m          # Scalar * Matrix
```

#### Matrix Multiplication

```python
result = m @ m2         # Matrix @ Matrix
```
- Follows standard matrix multiplication rules.

#### Power

```python
result = m ** 3         # m raised to power 3 (m @ m @ m)
```

### Indexing and Assignment

```python
row = m[0]             # Get first row
m[1] = [9, 10]         # Set second row
```

### Properties and Operations

- **Flatten:** `m.flatten()` → Returns all elements as a flat list.
- **Transpose:** `m.transpose()` → Returns transposed matrix.
- **Trace:** `m.trace()` → Returns the sum of diagonal elements.
- **Norm:** `m.norm(type="frobenius")` → Returns Frobenius norm (default).
- **Elementwise Exponentiation:** `m.elementwise_exp(exp)` → Raises each element to `exp`.
- **Duplicate:** `m.duplicate()` → Returns a deep copy.
- **Shape/Dimensions:** `m.shape` or `m.dimensions` → `(rows, cols)` or string.
- **Square:** `m.is_square()` → Returns True if square.
- **Symmetric:** `m.is_symmetric()` → Returns True if symmetric.
- **Orthogonal:** `m.is_orthogonal()` → Returns True if orthogonal.

### Class Methods

- **Zeroes:** `Matrix.zeroes(rows, cols, fill=0)` → Matrix filled with `fill`.
- **Identity:** `Matrix.identity(n)` → n×n identity matrix.
- **Diagonal:** `Matrix.diagonal_matrix(list)` → Diagonal matrix from list.
- **Random:** `Matrix.rnd_matrix(rows, cols, min=0, max=10)` → Random integer matrix.

---

## Fractions Class

A custom fraction class for rational arithmetic, supporting:
- Integer, float, and nested fraction inputs
- Arithmetic (+, -, *, /, %, **)
- Comparison (==, !=, <, >, <=, >=)
- Reduction to lowest terms
- Mixed number conversion, reciprocal, sign
- Special values (zero, one, half, infinity, nan)
- Newton-Raphson square root approximation

### Construction

```python
f = Fractions(3, 4)          # 3/4
f2 = Fractions(2.5, 7)       # Handles floats (converted to int)
f3 = Fractions(f, 2)         # Nested fractions (flattened)
```
- Accepts integers, floats, or Fractions as numerator/denominator.
- Autoreduces if `Fractions.autoreduce = True`.

### Display

```python
print(f)                     # e.g. 3⁄4 (Unicode slash)
```

### Arithmetic

```python
f + f2                       # Add
f - f2                       # Subtract
f * 3                        # Multiply by integer
3 * f                        # Integer * Fraction
f / f2                       # Divide
f % f2                       # Modulus
f ** 2                       # Power
-f                           # Negation
abs(f)                       # Absolute value
```

### Comparison

```python
f == f2
f != f2
f < f2
f > f2
f <= f2
f >= f2
```

### Reduction and Mixed Numbers

- **Reduce:** `f.reduce()` → Reduces to lowest terms.
- **Check Reducible:** `f.reducible()`
- **Is Reduced:** `f.is_reduced()`
- **To Mixed Number:** `f.to_mixed_number(intrem=False)` → Returns (integer, fractional part)
- **Reciprocal:** `f.reciprocal()`
- **Sign:** `f.sign()` → 1, -1, or 0

### Special Class Methods

- **Zero:** `Fractions.zero()` → 0/1
- **One:** `Fractions.one()` → 1/1
- **Half:** `Fractions.half()` → 1/2
- **Infinity:** `Fractions.infinity(sign=1)` → 1/0 or -1/0
- **NaN:** `Fractions.nan()` → 0/0

### Advanced: Newton-Raphson Square Root Approximation

```python
Fractions.newtonraphson(radicand, iterations, y0_1=False)
```
- Approximates √radicand as a fraction using Newton-Raphson method.
- Useful for rational approximations of irrational roots.

---

## Example Usage

```python
from grayscale_lib import Matrix, Fractions

# Matrix operations
A = Matrix([[1, 2], [3, 4]])
B = Matrix.identity(2)
C = Matrix.rnd_matrix(2, 2, 0, 5)
print(A + B)
print(A @ B)
print(A.transpose())

# Fractions operations
f1 = Fractions(1, 3)
f2 = Fractions(2, 5)
print(f1 + f2)
print(f1 * 4)
mixed = f1.to_mixed_number()
print(mixed)
sqrt_approx = Fractions.newtonraphson(2, 5)
print(sqrt_approx)
```

---

## Notes

- All operations raise appropriate errors for invalid inputs (e.g., mismatched dimensions for matrix arithmetic).
- Unicode formatting may not display in all environments/fonts.
- The Fractions class does not support complex numbers.
- Some advanced matrix operations (e.g. determinant, inverse) are **not implemented**.
- The `from_string` method in Fractions is incomplete and commented out.

---

## License

This code is provided by [Greygar724665](https://github.com/Greygar724665) for educational and utility purposes. See repository for license information.

---