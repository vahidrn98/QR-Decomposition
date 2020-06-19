import math

def zeros(m,n):

    return [[0 for i in range(n)] for j in range(m)]

def zeros_single(m):

    return [0 for j in range(m)]

def norm(A):

    return math.sqrt(sum([pow(x,2) for x in A]))

def transpose(A,m,n):
    
    return [[A[i][j] for i in range(m)] for j in range(n)]

def column(A,i):

    return [row[i] for row in A]

def vec_mult(A,B):

    return sum([A[i]*B[i] for i in range(len(A))])

def mult(A,x):

    return [x*A[i] for i in range(len(A))]

def mult_matrix(A,B):

    return [vec_mult(A[i],B) for i in range(len(A))]

def mat_mult(A,B):

    return [[vec_mult(A[i],column(B,j)) for i in range(len(A))] for j in range(len(B[0]))]

def main():

    with open('new.txt','r') as f:

        n_cases = int(f.readline())
        for case in range(n_cases):
            print("test case # ",case+1)
            n = int(f.readline())
            m = int(f.readline())
            A = zeros(n,m)
            y = zeros_single(n)
            x = zeros_single(m)
            Q_t = zeros(m,n)
            R = zeros(m,m)
            v = zeros_single(n)

            for i in range(n):
                for j in range(m):
                    A[i][j] = float(f.readline())

            for i in range(n):
                y[i] = float(f.readline())
            
            for k in range(m):
                v = column(A,k)
                tmp_v =v

                for i in range(k):
                    # print(len(Q_t),' ',i)
                    R[i][k] = vec_mult(Q_t[i],column(A,k))

                    for j in range(len(tmp_v)):
                        tmp_v[j] = tmp_v[j] - mult(Q_t[i],R[i][k])[j]

                R[k][k] = norm(v)
                Q_t[k] = [x/R[k][k] for x in v]

            Q = transpose(Q_t,m,n)

            print(A)

            print(mat_mult(Q,R))
            
            ##### back substitution #####

            b = mult_matrix(Q_t,y)

            with open('output.txt','a') as o:
                for k in reversed(range(m)):
                    x[k] = (b[k]+sum([-1*R[k][i]*x[i] for i in reversed(range(k)) ]))/R[k][k]
                    o.write(str(x[k])+'\n')


            print(max([vec_mult(A[i],x)-y[i] for i in range(n)]))

            
main()
# print(mat_mult([[2,2],[3,3],[3,3]],[[2,2,1],[3,3,2]]))
