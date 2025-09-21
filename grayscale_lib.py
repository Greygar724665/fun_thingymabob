from math import sqrt, gcd, lcm, floor
from random import randint
class Matrix:
    #region Magic Methods
    """
    A class to represent a mathematical matrix and perform matrix operations.
    Supports addition, subtraction, scalar multiplication, matrix multiplication,
    power, transpose, trace, norm, and various matrix property checks.
    """
    def __init__(self, data):
        """
        Initialize a Matrix object.
        :param data: List of lists representing matrix rows.
        """
        self.rows=len(data) # Number of rows
        self.cols=len(data[0]) # Number of columns
        for i in range(1,self.rows): # Check whether all rows have the same number of columns.
            if len(data[i]) != self.cols:
                raise ValueError("All rows must have the same number of columns")
        self.dimensions=f"{self.rows}\u00D7{self.cols}" # Dimensions written like r×c
        self.data = data
        self.shape = (self.rows, self.cols)
    
    def __str__(self):
        """
        Return a matrix-like string representation using Unicode symbols.
        """
        col_widths = [max(len(str(self.data[r][c])) for r in range(self.rows)) for c in range(self.cols)]
        matrix_lines = []
        for r in range(self.rows):
            row_items = [str(self.data[r][c]).rjust(col_widths[c]) for c in range(self.cols)]
            line = " ".join(row_items)
            if r == 0:
                matrix_lines.append(f"⎡ {line} ⎤")
            elif r == self.rows - 1:
                matrix_lines.append(f"⎣ {line} ⎦")
            else:
                matrix_lines.append(f"⎥ {line} ⎥")
        return "\n".join(matrix_lines)
    
    def __eq__(self, other):
        """
        Check if two matrices are equal (elementwise).
        """
        if self.data == other.data:
            return True
        return False

    def __add__(self, other):
        # Explanation of the mathematical process:
        # To add two matrices, add corresponding elements from each matrix.
        # For matrix and scalar, add the scalar to each element.
        """
        Add two matrices or a matrix and a scalar.
        """
        sumMatrix=[[0] * self.cols for i in range(self.rows)]
        if isinstance(other, type(self)):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrices must have the same dimensions in addition")
            for i in range(self.rows):
                for j in range(self.cols):
                    sumMatrix[i][j]=(self.data[i][j])+other.data[i][j]
            return Matrix(sumMatrix)
        if isinstance(other, (int, float)):
            for i in range(self.rows):
                for j in range(self.cols):
                    sumMatrix[i][j]=self.data[i][j]+other
            return Matrix(sumMatrix)
    
    def __radd__(self, other):
        # Explanation of the mathematical process:
        # Add a scalar to each element of the matrix (right-hand operation).
        """
        Add a scalar to a matrix (right-hand).
        """
        sumMatrix=[[0] * self.cols for i in range(self.rows)]
        for i in range(self.rows):
                for j in range(self.cols):
                    sumMatrix[i][j]=self.data[i][j]+other
        return Matrix(sumMatrix)

    def __sub__(self, other):
        # Explanation of the mathematical process:
        # To subtract two matrices, subtract corresponding elements from each matrix.
        # For matrix and scalar, subtract the scalar from each element.
        """
        Subtract two matrices or a scalar from a matrix.
        """
        diffMatrix=[[0] * self.cols for i in range(self.rows)]
        if isinstance(other, type(self)):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrices must have the same dimensions in subtraction")
            for i in range(self.rows):
                for j in range(self.cols):
                    diffMatrix[i][j]=(self.data[i][j])-other.data[i][j]
            return Matrix(diffMatrix)
        elif isinstance(other, (int, float)):
            for i in range(self.rows):
                for j in range(self.cols):
                    diffMatrix[i][j]=self.data[i][j]-other
        return Matrix(diffMatrix)

    def __rsub__(self,other):
        # Explanation of the mathematical process:
        # Subtract each element of the matrix from the scalar (right-hand operation).
        """
        Subtract a matrix from a scalar (right-hand).
        """
        diffMatrix=[[0] * self.cols for i in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                diffMatrix[i][j]=other-self.data[i][j]
        return Matrix(diffMatrix)

    def __mul__(self, scalar):
        # Explanation of the mathematical process:
        # Multiply each element of the matrix by the scalar.
        """
        Multiply a matrix by a scalar.
        """
        s_prodMatrix=[[0] * self.cols for i in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                s_prodMatrix[i][j]=self.data[i][j]*scalar
        return Matrix(s_prodMatrix)
    
    def __rmul__(self, other):
        # Explanation of the mathematical process:
        # Multiply each element of the matrix by the scalar (right-hand operation).
        """
        Multiply a scalar by a matrix (right-hand).
        """
        return self*other
    
    def __pow__(self, scalar):
        # Explanation of the mathematical process:
        # Raise the matrix to a power by repeated matrix multiplication.
        """
        Raise a matrix to a power (repeated matrix multiplication).
        """
        i=0
        matrixCopy=self.duplicate()
        while i<scalar-1:
            matrixCopy@=self
            i+=1
        return matrixCopy

    def __matmul__(self,other):
        # Explanation of the mathematical process:
        # Matrix multiplication: For each element (i, j) in the result,
        # take the dot product of row i from the first matrix and column j from the second matrix.
        """
        Matrix multiplication using the @ operator.
        """
        if self.cols != other.rows:
            raise ValueError("Number of colums of the first matrix have to be equal to the number of rows of the second matrix.")
        m_prodMatrix=[[0] * other.cols for i in range(self.rows)]
        # return Matrix(m_prodMatrix)
        for row in range(self.rows):
            for col in range(other.cols):
                sum=0
                for k in range(self.cols):
                    sum+=self.data[row][k]*other.data[k][col]
                m_prodMatrix[row][col]=sum
        return Matrix(m_prodMatrix)
    
    def __getitem__(self, index):
        """
        Get a row of the matrix by index.
        :param index: Row index.
        :return: List representing the row at the given index.
        """
        return self.data[index]
    
    def __setitem__(self, index, value):
        """
        Set a row of the matrix by index.
        :param index: Row index.
        :param value: List to assign to the row.
        :raises ValueError: If value is not a list of correct length.
        """
        if isinstance(value, list) and len(value) == self.cols:
            self.data[index] = value
        else:
            raise ValueError("Assigned value must be a list with the correct number of columns")
    #endregion Magic Methods
    
    #region Instance Methods
    def __len__(self):
        """
        Return the number of rows in the matrix.
        :return: Number of rows.
        """
        return self.rows
    
    def flatten(self):
        return [elem for row in self.data for elem in row]
    
    def transpose(self):
        """
        Return the transpose of the matrix.
        """
        transposed= [[0] * self.rows for i in range(self.cols)]
        for i in range(self.cols):
            for j in range(self.rows):
                transposed[i][j]=self.data[j][i]
        return Matrix(transposed)
    
    def trace(self):
        """
        Return the trace (sum of diagonal elements) of the matrix.
        """
        sum=0
        for i in range(self.rows):
            sum+=self.data[i][i]
        return sum
    
    def norm(self, type="frobenius"):
        """
        Calculate the norm of the matrix. Default is Frobenius norm.
        """
        if type=="frobenius":
            sum=0
            for i in range(self.rows):
                for j in range(self.cols):
                    sum+=(self.data[i][j])**2
            return sqrt(sum)
        
        # elif type=="1norm":
        #     for i in range(self.cols):
    
    def elementwise_exp(self, exponent):
        """
        Raise each element of the matrix to the given exponent.
        """
        expMatrix=[[0] * self.cols for i in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                expMatrix[i][j]=self.data[i][j]**exponent
        return Matrix(expMatrix)
    
    def duplicate(self):
        """
        Return a deep copy of the matrix.
        """
        return Matrix([row[:] for row in self.data])

    def is_square(self):
        """
        Check if the matrix is square.
        """
        if self.rows == self.cols:
            return True
        return False
    
    def is_symmetric(self):
        """
        Check if the matrix is symmetric.
        """
        if self == self.transpose():
            return True
        return False
    #endregion Instance Methods
    
    #region Class Methods
    def is_orthogonal(self):
        """
        Check if the matrix is orthogonal.
        """
        if not self.is_square():
            return False
        return (self@self.transpose() == Matrix.identity(self.rows))
    
    @classmethod
    def zeroes(cls, rows, cols, fill=0):
        """
        Create a matrix of given size filled with a specified value.
        """
        if rows == 0 or cols == 0:
            raise ValueError("Rows and/or columns can not be 0")
        if isinstance(fill, (int, float)):
            return Matrix([[fill] * cols for i in range(rows)])
        raise TypeError("Fill of the zeroes function must be either an integer or a float")
    @classmethod
    def identity(cls, n):
        """
        Create an identity matrix of size n x n.
        """
        if n==0:
            raise ValueError("Identity dimensions can not be 0")
        raw=[[0] * n for i in range(n)]
        for i in range(n):
            raw[i][i]=1
        return Matrix(raw)
    @classmethod
    def diagonal_matrix(cls, list):
        """
        Create a diagonal matrix from a list of values.
        Each value will be placed on the diagonal, with zeroes elsewhere.
        :param list: List of values for the diagonal.
        :return: Matrix with the given diagonal values.
        """
        n = len(list)
        newMatrix = cls.zeroes(n, n)
        for i in range(n):
            newMatrix[i][i] = list[i]
        return newMatrix
    @classmethod
    def rnd_matrix(cls, r, c, min=0, max=10):
        """
        Create a matrix of size r x c filled with random integers between min and max (inclusive).
        :param r: Number of rows.
        :param c: Number of columns.
        :param min: Minimum random value (inclusive).
        :param max: Maximum random value (inclusive).
        :return: Matrix with random integer values.
        """
        newMatrix = cls.zeroes(r, c)
        for i in range(r):
            for j in range(c):
                newMatrix[i][j] = randint(min, max)
        return newMatrix

    #endregion Class Methods


class Fractions:
    #region Magic Methods
    autoreduce = False  # Turns off/on autoreduction for all Fractions
    def __init__(self, numerator, denominator=1):
        """
        Initialize a Fractions object.
        :param numerator: The numerator of the fraction (int or float).
        :param denominator: The denominator of the fraction (int or float, defaults to 1).
        :raises TypeError: If numerator or denominator is complex.
        :raises ZeroDivisionError: If denominator is zero.
        """
        # Store numerator and denominator
        self.numerator = numerator
        self.denominator = denominator
        # Explanation: Fractions cannot have complex numbers as numerator or denominator.
        if isinstance(self.numerator, complex) or isinstance(self.denominator, complex):
            raise TypeError("This fractions class does not support complex numbers.")
        # Explanation: Convert whole numbers to integers for consistency.
        if self.numerator % 1 == 0 and self.denominator % 1 == 0:
            self.numerator = int(self.numerator)
            self.denominator = int(self.denominator)
        # Explanation: If either term is a float, convert both to float and normalize decimal places.
        if isinstance(self.numerator, float) or isinstance(self.denominator, float):
            self.numerator = float(self.numerator)
            self.denominator = float(self.denominator)
            # Count decimal places
            strNumer = str(self.numerator)
            strDenom = str(self.denominator)
            numerDot = strNumer.find(".")
            denomDot = strDenom.find(".")
            decimalsNumer = strNumer[numerDot+1:]
            decimalsDenom = strDenom[denomDot+1:]
            countNumer = len(decimalsNumer)
            countDenom = len(decimalsDenom)
            if decimalsNumer == "0":
                countNumer = 0
            if decimalsDenom == "0":
                countDenom = 0
            # Multiply both terms by 10 raised to the number of decimal places in the term with the most decimal places
            if countDenom >= countNumer:
                self.numerator = int(self.numerator * (10 ** countDenom))
                self.denominator = int(self.denominator * (10 ** countDenom))
            else:
                self.numerator = int(self.numerator * (10 ** countNumer))
                self.denominator = int(self.denominator * (10 ** countNumer))
        # Explanation: Normalize sign so denominator is always positive
        if self.denominator < 0:
            self.numerator = -self.numerator
            self.denominator = abs(self.denominator)
        # Explanation: Reduce the fraction if autoreduce is enabled and denominator is not 1
        if Fractions.autoreduce and self.denominator != 1 and self.reducible():
            gtCD = gcd(self.numerator, self.denominator)
            self.numerator //= gtCD
            self.denominator //= gtCD

    def __str__(self):
        """
        Return string representation of the fraction using Unicode fraction slash.
        :return: String in the form 'numerator⁄denominator'.
        """
        # Explanation: Unicode fraction slash may not be supported in all fonts.
        return f"{self.numerator}\u2044{self.denominator}"
    
    def __eq__(self, other):
        """
        Check equality between this fraction and another fraction or integer.
        :param other: Fractions object or integer.
        :return: True if equal, False otherwise.
        """
        if isinstance(other, int):
            other = Fractions(other, 1)
        if isinstance(other, Fractions):
            return self.numerator * other.denominator == other.numerator * self.denominator
        return False

    def __add__(self, other):
        # Convert integer to fraction
        # This is done to ensure that we can add fractions and integers seamlessly.
        if isinstance(other, int):
            other = Fractions(other, 1)

        if isinstance(other, Fractions):
            # Explanation of the mathematical process via words:
            # To add two fractions, we find a common denominator by multiplying the denominators of both fractions.
            # We then adjust the numerators accordingly and add them to get the new numerator.
            new_den = self.denominator * other.denominator
            new_num = (self.numerator * other.denominator) + (other.numerator * self.denominator)
            return Fractions(new_num, new_den)
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        # Convert integer to fraction
        if isinstance(other, int):
            other = Fractions(other, 1)
        # Explanation of the mathematical process via words:
        # To subtract two fractions, we find a common denominator by multiplying the denominators of both fractions.
        # We then adjust the numerators accordingly and subtract them to get the new numerator.
        if isinstance(other, Fractions):
            new_den = self.denominator * other.denominator
            new_num = (self.numerator * other.denominator) - (other.numerator * self.denominator)
            return Fractions(new_num, new_den)
    
    def __rsub__(self, other):
        """
        Right-hand subtraction for Fractions.
        :param other: Integer or Fractions object.
        :return: Fractions object representing the difference.
        """
        return Fractions(other,1) - self
    
    def __mul__(self, other):
        # Explanation of the mathematical process via words(integer multiplicand):
        # To multiply a fraction by an integer, we multiply the numerator by the integer.
        if isinstance(other, int):
            return Fractions(self.numerator*other, self.denominator)
        # Explanation of the mathematical process via words(fraction multiplicand):
        # To multiply two fractions, we multiply the numerators together to get the new numerator,
        # and we multiply the denominators together to get the new denominator.
        if isinstance(other, Fractions):
            return Fractions(self.numerator*other.numerator, self.denominator*other.denominator)
    
    def __rmul__(self, other):
        """
        Right-hand multiplication for Fractions.
        :param other: Integer or Fractions object.
        :return: Fractions object representing the product.
        """
        return self * other
    
    def __truediv__(self, other):
        # Explanation of the mathematical process via words:
        # To divide two fractions, we multiply the numerator of the first fraction by the denominator of the second fraction,
        # and we multiply the denominator of the first fraction by the numerator of the second fraction.
        return self*Fractions(other.denominator, other.numerator)
    
    def __rtruediv__(self, other):
        """
        Right-hand division for Fractions.
        :param other: Integer or Fractions object.
        :return: Fractions object representing the quotient.
        """
        return Fractions(other,1)/self
    
    def __mod__(self, other):
        # Explanation of the mathematical process via words:
        # To find the modulus of two fractions, we multiply the numerator of the first fraction by the denominator of the second fraction,
        # and then take the result modulo the product of the numerator of the second fraction and the denominator of the first fraction.
        return Fractions((self.numerator*other.denominator)%(other.numerator*self.denominator), (self.denominator*other.denominator))

    def __rmod__(self, other):
        """
        Right-hand modulus for Fractions.
        :param other: Integer or Fractions object.
        :return: Fractions object representing the modulus.
        """
        # Explanation of the mathematical process via words(right-hand operation):
        # To find the modulus of a fraction and an integer, we multiply the fraction by the integer's denominator,
        # and then take the result modulo the product of the integer's numerator and the fraction's denominator.
        return other-Fractions(floor((other/self).eval))*self
    
    def __pow__(self, exponent):
        """
        Raise the fraction to a power.
        :param exponent: The exponent to raise to.
        :return: Fractions object representing the result.
        """
        # Explanation of the mathematical process via words:
        # To raise a fraction to a power, we raise both the numerator and the denominator to that power.
        return Fractions(self.numerator**exponent, self.denominator**exponent)

    def __neg__(self):
        """
        Return the negation of the fraction.
        :return: Fractions object representing the negated value.
        """
        # Explanation: Negating a fraction flips the sign of the numerator, keeping denominator positive.
        return Fractions(-self.numerator, self.denominator)
    
    def __hash__(self):
        """
        Return a hash value for the fraction.
        The hash is based on the numerator and denominator.
        Warning: The fraction is reduced before hashing to ensure consistent hash values for equivalent fractions.
        :return: Hash value.
        """
        self.reduce()
        return hash((self.numerator, self.denominator))

    #endregion Magic Methods

    #region Instance Methods
    def __abs__(self):
        """
        Return the absolute value of the fraction.
        :return: Fractions object representing the absolute value.
        """
        return Fractions(abs(self.numerator), abs(self.denominator))
    
    def reducible(self):
        """
        Check if the fraction can be reduced.
        :return: True if reducible, False otherwise.
        """
        return not self.is_reduced()
    
    def is_reduced(self):
        """
        Check if the fraction is already reduced.
        :return: True if reduced, False otherwise.
        """
        return gcd(self.numerator, self.denominator) == 1
    
    def reduce(self):
        """
        Reduce the fraction to its lowest terms.
        """
        # Explanation of the mathematical process via words:
        # To reduce a fraction, we find the greatest common divisor (GCD) of the numerator and denominator.
        # We then divide both the numerator and denominator by the GCD to get the reduced fraction
        gtCD = gcd(self.numerator, self.denominator)
        self.numerator //= gtCD
        self.denominator //= gtCD
    
    def reciprocal(self):
        """
        Return the reciprocal of the fraction.
        :return: Fractions object representing the reciprocal.
        """
        return Fractions(self.denominator, self.numerator)
    
    def to_mixed_number(self, intrem=False):
        """
        Return the mixed number representation of the fraction.
        :param intrem: If True, returns integer and remainder. If False, returns integer and fractional part.
        :return: Tuple (integer part, remainder or fractional part).
        """
        if not intrem:
            return (self.numerator//self.denominator, Fractions(self.numerator%self.denominator, self.denominator))
        return (self.numerator//self.denominator, self.numerator%self.denominator)
    
    def sign(self):
        """
        Return the sign of the fraction.
        :return: 1 if positive, -1 if negative, 0 if zero.
        """
        if self.numerator == 0:
            return 0
        absolute = abs(self.numerator)
        if absolute != self.numerator:
            return -1
        elif absolute == self.numerator:
            return 1
    #endregion Instance Methods

    #region Class Methods

    @classmethod
    def zero(cls):
        """
        Return a fraction representing zero.
        """
        return Fractions(0,1)
    @classmethod
    def one(cls):
        """
        Return a fraction representing one.
        """
        return Fractions(1,1)
    @classmethod
    def half(cls):
        """
        Return a fraction representing one half.
        """
        return Fractions(1,2)
    @classmethod
    def infinity(cls, sign=1):
        """
        Return a fraction representing infinity (with sign).
        :param sign: 1 for positive infinity, -1 for negative infinity.
        """
        if sign in [-1,1]:
            return Fractions(sign,0)
    @classmethod
    def nan(cls):
        """
        Return a fraction representing NaN (not a number).
        """
        return Fractions(0,0)

    #endregion Class Methods

    # The from_string method is commented out because it is incomplete and does not handle all cases.
    # It needs further development to correctly parse strings into Fractions.
    # @classmethod
    # def from_string(cls, string):
    #     # s contains none of the substrings
    #     div_symbols = "÷∕⁄/⨸➗⊘⦼"
    #     allowed_symbols = div_symbols + "0123456789.-"
    #     if any(char not in allowed_symbols for char in string):
    #         raise ValueError("The string contains invalid characters. Only digits, decimal points, negative signs, and division symbols are allowed.")
            
    #     # Find the first occurrence of any symbol
    #     indices = [string.find(sym) for sym in div_symbols if sym in string]
        
    #     # Check that checks if the string only contains a number
    #     if all(char in "0123456789.-" for char in string):
    #         return Fractions(float(string), 1)
        
    #     # Explain what the code does:
    #     # The code searches for the first occurrence of any division symbol in the input string.
    #     if indices:
    #         first_index = min(indices)
    #     first_index