from math import sqrt

class Matrix:
    """
    A class to represent a mathematical matrix and perform matrix operations.
    Supports addition, subtraction, scalar multiplication, matrix multiplication,
    power, transpose, trace, norm, and various matrix property checks.
    """
    #region Instance Methods
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
        Return a pretty-printed string representation of the matrix.
        """
        # find the max width of each column
        col_widths = [max(len(str(self.data[r][c])) for r in range(self.rows))
                      for c in range(self.cols)]

        # build rows with proper spacing
        rows_str = []
        for r in range(self.rows):
            row_items = [str(self.data[r][c]).rjust(col_widths[c]) for c in range(self.cols)]
            rows_str.append(" | ".join(row_items))
        return ("\n".join(rows_str))+f"\n{self.dimensions}\n"
    
    def __eq__(self, other):
        """
        Check if two matrices are equal (elementwise).
        """
        if self.data == other.data:
            return True
        return False

    def __add__(self, other):
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
        """
        Add a scalar to a matrix (right-hand).
        """
        sumMatrix=[[0] * self.cols for i in range(self.rows)]
        for i in range(self.rows):
                for j in range(self.cols):
                    sumMatrix[i][j]=self.data[i][j]+other
        return Matrix(sumMatrix)

    def __sub__(self, other):
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
        """
        Subtract a matrix from a scalar (right-hand).
        """
        diffMatrix=[[0] * self.cols for i in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                diffMatrix[i][j]=other-self.data[i][j]
        return Matrix(diffMatrix)

    def __mul__(self, scalar):
        """
        Multiply a matrix by a scalar.
        """
        s_prodMatrix=[[0] * self.cols for i in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                s_prodMatrix[i][j]=self.data[i][j]*scalar
        return Matrix(s_prodMatrix)
    
    def __rmul__(self, other):
        """
        Multiply a scalar by a matrix (right-hand).
        """
        return self*other
    
    def __pow__(self, scalar):
        """
        Raise a matrix to a power (repeated matrix multiplication).
        """
        i=0
        matrixCopy=self.duplicate()
        while i<scalar-1:
            matrixCopy@=self
            i+=1

    def __matmul__(self,other):
        """
        Matrix multiplication using the @ operator.
        """
        if self.cols != other.rows:
            raise ValueError("Number of colums of the first matrix have to be equal to the number of rows of the second matrix.")
        m_prodMatrix=[[0] * self.rows for i in range(other.cols)]
        # return Matrix(m_prodMatrix)
        for row in range(self.rows):
            for col in range(other.cols):
                sum=0
                for k in range(self.cols):
                    sum+=self.data[row][k]*other.data[k][col]
                m_prodMatrix[row][col]=sum
        return Matrix(m_prodMatrix)
    
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
    
    def is_orthogonal(self):
        """
        Check if the matrix is orthogonal.
        """
        if not self.is_square():
            return False
        return (self@self.transpose() == Matrix.identity(self.rows))
    
    #endregion Instance Methods
    #region Class Methods
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
    #endregion Class Methods


