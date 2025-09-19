from math import sqrt

class Matrix:
    #region Instance Methods
    def __init__(self, data):
        self.rows=len(data) # Number of rows
        self.cols=len(data[0]) # Number of columns
        for i in range(1,self.rows): # Check whether all rows have the same number of columns.
            if len(data[i]) != self.cols:
                raise ValueError("All rows must have the same number of columns")
        self.dimensions=f"{self.rows}\u00D7{self.cols}" # Dimensions written like r×c
        self.data = data
        self.shape = (self.rows, self.cols)
    
    def __str__(self):
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
        if self.data == other.data:
            return True
        return False

    def __add__(self, other):
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
        sumMatrix=[[0] * self.cols for i in range(self.rows)]
        for i in range(self.rows):
                for j in range(self.cols):
                    sumMatrix[i][j]=self.data[i][j]+other
        return Matrix(sumMatrix)

    def __sub__(self, other):
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
        diffMatrix=[[0] * self.cols for i in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                diffMatrix[i][j]=other-self.data[i][j]
        return Matrix(diffMatrix)

    def __mul__(self, scalar):
        s_prodMatrix=[[0] * self.cols for i in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                s_prodMatrix[i][j]=self.data[i][j]*scalar
        return Matrix(s_prodMatrix)
    
    def __rmul__(self, other):
        return self*other
    
    def __pow__(self, scalar):
        resultMatrix=[[0] * self.cols for i in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                resultMatrix[i][j]=self.data[i][j]**scalar
        return Matrix(resultMatrix)
    
    def __matmul__(self,other):
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
        transposed= [[0] * self.rows for i in range(self.cols)]
        for i in range(self.cols):
            for j in range(self.rows):
                transposed[i][j]=self.data[j][i]
        return Matrix(transposed)
    
    def trace(self):
        sum=0
        for i in range(self.rows):
            sum+=self.data[i][i]
        return sum
    
    def norm(self, type="frobenius"):
        if type=="frobenius":
            sum=0
            for i in range(self.rows):
                for j in range(self.cols):
                    sum+=(self.data[i][j])**2
            return sqrt(sum)
        
        # elif type=="1norm":
        #     for i in range(self.cols):
    
    def is_square(self):
        if self.rows == self.cols:
            return True
        return False
    
    def is_symmetric(self):
        if self == self.transpose():
            return True
        return False
    
    def is_orthogonal(self):
        if not self.is_square():
            return False
        return (self@self.transpose() == Matrix.identity(self.rows))
    #endregion Instance Methods
    #region Class Methods
    @classmethod
    def zeroes(cls, rows, cols, fill=0):
        if rows == 0 or cols == 0:
            raise ValueError("Rows and/or columns can not be 0")
        if isinstance(fill, (int, float)):
            return Matrix([[fill] * cols for i in range(rows)])
        raise TypeError("Fill of the zeroes function must be either an integer or a float")
    @classmethod
    def identity(cls, n):
        if n==0:
            raise ValueError("Identity dimensions can not be 0")
        raw=[[0] * n for i in range(n)]
        for i in range(n):
            raw[i][i]=1
        return Matrix(raw)
    #endregion Class Methods



