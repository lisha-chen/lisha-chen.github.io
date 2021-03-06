---
layout:     post
title:      How to align a face
subtitle:   Affine transformation
date:       2019-01-31
author:     Lisha Chen
header-img: img/post-bg-chalkboard.png
catalog: true
tags:
    - computer vision
    - geometry
    - linear algebra

---


## Introduction

This post aims to provide different ways to align frontal or near-frontal faces based on facial landmarks. To align a face, the goal is to do geometric transformation to the image to make certain facial landmarks, eg. eye corners, nose tip located at the target location. Typical transformations we apply to the image includes rotation, translation, scaling, or general affine transformation.

Aligning faces typically serves as a preprocessing step for higher-level facial behavior analysis tasks, eg. face recognition, facial action unit recognition, facial expression recognition.

## Different transformations

We use homogeneous coordinates $$\mathbf{p} = \begin{bmatrix}x &y &1\end{bmatrix}^T$$.


### Translation

Translation needs two parameters $$t_x, t_y$$.
$$
T = \begin{bmatrix} 1 & 0 & t_x\\ 
0 & 1 & t_y\\
0 &0 &1
\end{bmatrix}
$$


### Rotation

In-plane rotation with zero center needs one parameter $$\theta$$.
$$
R_0 = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ 
\sin\theta & \cos\theta & 0 \\
0 &0 &1 \\
\end{bmatrix}
$$

Rotation with non-zero center
$$
R = \begin{bmatrix} \cos\theta & -\sin\theta & x_{r0}(1-\cos\theta)+ y_{r0}\sin\theta \\
\sin\theta & \cos\theta &- x_{r0}\sin\theta+ y_{r0}(1-\cos\theta)\\
0 &0 &1
\end{bmatrix}
$$

Rotation with non-zero center can be decomposed as

$$
\begin{align}
R &= \begin{bmatrix} \cos\theta & -\sin\theta & x_{r0}(1-\cos\theta)+ y_{r0}\sin\theta \\ 
\sin\theta & \cos\theta &- x_{r0}\sin\theta+ y_{r0}(1-\cos\theta)\\
0 &0 &1
\end{bmatrix}\\
&= \begin{bmatrix} 1 & 0 & x_{r0}(1-\cos\theta)+ y_{r0}\sin\theta \\ 
0 & 1 & - x_{r0}\sin\theta+ y_{r0}(1-\cos\theta)\\
0 &0 &1
\end{bmatrix}
\begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ 
\sin\theta & \cos\theta & 0\\
0 &0 &1
\end{bmatrix}\\
&= T_RR_0
\end{align}
$$


### Scaling

Non-uniform scaling with zero center needs two parameters $$s_x,s_y$$.
$$
\begin{equation}
S_0 = \begin{bmatrix}
s_x & 0 & 0\\ 
0 & s_y & 0\\
0 &0 &1
\end{bmatrix}
\end{equation}
$$

Non-uniform scaling in directions other than vertical and horizontal may need to combine rotations with $$S_0$$.

Scaling with non-zero center $$[x_{s0},y_{s0}]$$.
$$
S = \begin{bmatrix} 
s_x & 0 & x_{s0}(1-s_x)\\ 
0 & s_y & y_{s0}(1-s_y)\\
0 &0 &1
\end{bmatrix}
$$

For uniform scaling $$S_{u0}, S_u$$ , $$s_x=s_y$$.

Scaling with non-zero center can be decomposed as a scaling with zero center and a translation.

$$
S = \begin{bmatrix} s_x & 0 & x_{s0}(1-s_x)\\ 
0 & s_y & y_{s0}(1-s_y)\\
0 &0 &1
\end{bmatrix}
= \begin{bmatrix} 1 & 0 & x_{s0}(1-s_x)\\ 
0 & 1 & y_{s0}(1-s_y)\\
0 &0 &1
\end{bmatrix}
\begin{bmatrix} s_x & 0 & 0\\ 
0 & s_y & 0\\
0 &0 &1
\end{bmatrix}
=T_S S_0
$$

### Rotation+Scaling+Translation

One way of transforming the image without warping is to apply rotation, uniform scaling and translation.

$$
\begin{aligned}
\mathbf{p}^{rst}
&=TS_u R\mathbf{p}\\
&=T T_S S_{u0}T_R R_0 \mathbf{p}\\
&=T T_S T_{R'} S_{u0} R_0 \mathbf{p}\\
&=T_{tsr} S_{u0} R_0 \mathbf{p}\\
&=\begin{bmatrix}
s\cos\theta & -s\sin\theta & t_x \\ 
s\sin\theta & s\cos\theta & t_y\\
0 &0 &1
\end{bmatrix}
\end{aligned}
$$

To make $$T_{R'}S_0=S_0T_R$$, $$T_{R'}=S_0T_RS_0^{-1}$$, still a translation matrix.

There are totally 4 independent parameters in this transformation matrix, we need at least 2 points to solve the matrix.

If scaling is non-uniform, then there are totally 5 independent parameters in the transformation matrix and we need at least 3 points to solve the matrix.



### Affine transformation

The above transformation $$TSR$$ is just a special case of affine transformation.

$$
\mathbf{p}^{affine}=
\begin{bmatrix} a_{00} & a_{01} & b_{00}\\ 
a_{10} & a_{11} & b_{10}\\
0 &0 &1
\end{bmatrix}
\mathbf{p}
= A_{affine}\mathbf{p}
$$

$$
\begin{bmatrix}
\mathbf{p}_x^{affine}\\ 
\mathbf{p}_y^{affine}\\
\end{bmatrix}
=\begin{bmatrix} a_{00} & a_{01} \\ 
a_{10} & a_{11}\\
\end{bmatrix}
\begin{bmatrix}
\mathbf{p}_x\\ 
\mathbf{p}_y\\
\end{bmatrix}
+ 
\begin{bmatrix}
b_{00}\\
b_{10}\\
\end{bmatrix}
$$

Real matrix $$A$$ has real SVD (Since real symmetric matrix $$A^TA$$ has real eigenvalues $$\lambda$$, $$(A^TA-\lambda I)x=0$$ has real solution $$x$$. But $$U,V$$ are not unique, they can also be complex by applying conjugate complex signs to the column vectors of real $$U,V$$). $$A = \begin{bmatrix} a_{00} & a_{01} \\ 
a_{10} & a_{11}\\ \end{bmatrix} = U_A\Sigma_A V_A^T$$, it is equivalent to applying rotation/reflection with $$V_A^T$$, scaling with $$\Sigma_A$$ and rotation/reflection with $$U_A$$. Therefore an affine transformation is just different combinations of rotation, reflection, scaling, translation.

#### Properties preserved.

1. Collinearity & ratio of lengths.<br/> 
If $$\mathbf{p}_3 = \alpha \mathbf{p}_1 + (1-\alpha)\mathbf{p}_2 $$ , then $$A_{affine}\mathbf{p}_3 = \alpha A_{affine}\mathbf{p}_1 + (1-\alpha)A_{affine}\mathbf{p}_2 $$.
2. Parallelism.<br/>
If $$\mathbf{p}_2 = \alpha \mathbf{p}_1$$ , then $$A_{affine}\mathbf{p}_2 = \alpha A_{affine}\mathbf{p}_1 $$.
3. Convexity.<br/>
If $$ \mathbf{p}_3 = \alpha \mathbf{p}_1 + (1-\alpha)\mathbf{p}_2\in K $$, based on 1., If $$ A_{affine}\mathbf{p}_1 ,A_{affine}\mathbf{p}_2\in K_A=A_{affine}(K) $$, $$ A_{affine}\mathbf{p}_3 \in K_A $$.
4. Barycenters of weighted collections of points.<br/>
$$A_{affine}\sum_j w_j \mathbf{p}_j = \sum_j w_j A_{affine} \mathbf{p}_j $$.



## Align a face

To align the face using Affine transformation, we first solve the affine transformation matrix. Then we can do a backward mapping to find the corresponding point in the original image for each point in the target image and map the pixel values.

### Solve the transformation matrix

If we want to align 3 points such as two eye corners plus one nose tip, we directly solve the Affine transformation matrix which has 6 independent parameters and needs exactly 3 points.


We can use more points, eg. eye corners, nose tip, their current location in the original image and their target location in the transformed image. Then solve the linear system by least squares.

Let $$Vector(A_{affine}) = \begin{bmatrix}a_{00}&a_{01}&b_{00}&a_{10}&a_{11}&b_{10}\end{bmatrix}^T $$, the points before transformation $$\mathbf{p}_j$$, the points after transformation $$\mathbf{p}_j^a$$, $$j=1,\dots,N$$. We use homogeneous coordinates. Then the problem becomes

$$
\begin{bmatrix}
\mathbf{p}_{1x}^a\\
\mathbf{p}_{1y}^a\\
\vdots\\
\mathbf{p}_{Nx}^a\\
\mathbf{p}_{Ny}^a\\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{p}_{1}^T & \mathbf{0}\\
\mathbf{0} & \mathbf{p}_{1}^T\\
\vdots &\vdots\\
\mathbf{p}_{N}^T & \mathbf{0}\\
\mathbf{0} & \mathbf{p}_{N}^T\\
\end{bmatrix}
Vector(A_{affine})
$$

We use the normal equations to solve $$P^a = \mathbf{P} Vector(A_{affine})$$, then $$Vector(A_{affine}) = (\mathbf{P}^T\mathbf{P})^{-1}\mathbf{P}^T P^a$$.


### Impose orthogonal constraint on Affine transformation matrix

If we want to get uniform scaling, we need at least 2 points such as two eye corners. We could first compute rotation angle $$\theta$$ to make the line connecting two eye corners horizontal. Then we apply translation and scaling to make the two eye corners located at where we want in the target image. Or we can still first solve a linear system with orthogonal constraint and then obtain the rotation angle, scaling factor and translation.

If we impose the orthogonal constraint when solving the affine transformation matrix. The affine transformation becomes

$$\mathbf{p}^{affine}=
\begin{bmatrix}
a_{00} & -a_{10} & b_{00}\\ 
a_{10} & a_{00} & b_{10}\\
0 &0 &1
\end{bmatrix}
\mathbf{p}
= A_{affine}\mathbf{p}$$

Let $$Vector(A_{affine}) = \begin{bmatrix}a_{00} &a_{10} &b_{00} &b_{10}\end{bmatrix}^T $$, the problem becomes

$$
\begin{bmatrix}
\mathbf{p}_{1x}^a\\
\mathbf{p}_{1y}^a\\
\vdots\\
\mathbf{p}_{Nx}^a\\
\mathbf{p}_{Ny}^a\\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{p}_{1x} & -\mathbf{p}_{1y} & 1 & 0\\
\mathbf{p}_{1y} & \mathbf{p}_{1x} & 0 & 1\\
\vdots &\vdots &\vdots &\vdots\\
\mathbf{p}_{Nx} & -\mathbf{p}_{Ny} & 1 & 0\\
\mathbf{p}_{Ny} & \mathbf{p}_{Nx} & 0 & 1\\
\end{bmatrix}
Vector(A_{affine})
$$

Again we use the normal equations to solve $$P^a = \mathbf{P} Vector(A_{affine})$$, then $$Vector(A_{affine}) = (\mathbf{P}^T\mathbf{P})^{-1}\mathbf{P}^T P^a$$. Then $$s = \|[a_{00},a_{10}]\|_2$$ and $$\theta = sign(a_{10}) \arccos(\frac{a_{00}}{s})$$.



### Recover rotation and uniform scaling from general affine transformation matrix

After getting $$A_{affine}$$, if we want to recover rotation angle $$\theta$$ and uniform scaling factor $$s$$. We do the SVD of $$A=U_A\Sigma_AV_A^T$$ and tries to find the best approximating matrix $$\hat{s} U_A V_A^T$$ such that $$ \hat{s} = {\arg\min}_{s}\|sI - \Sigma_A\|$$, then the rotation matrix $$\hat{R}_0 = U_AV_A^T$$ and scaling factor $$s=\hat{s}$$ makes $$\|\hat{S}_{u0} \hat{R}_0 - A\| = \|U_A\hat{s}IV_A^T - U_A\Sigma_AV_A^T\| = \|\hat{s}I - \Sigma_A\|$$ minimized, $$\hat{s} = \frac{1}{2}trace(\Sigma_A)$$.

Note that 
1. $$\|\cdot\|$$ can be 2-norm or Frobenius norm. Typically we use Frobenius norm $$\|\cdot\| = \|\cdot\|_F$$.
2. $$\hat{R}_0 = U_AV_A^T$$ only imposes the orthogonal constraint, it doesn't make $$\hat{R}_0$$ a rotation matrix because an orthogonal matrix can also be a reflection matrix. Whether it is a rotation matrix depends on the points provided.

Another derivation to minimize the Frobenius norm without using SVD is as below

$$
\begin{aligned}
\hat{R}_0 
&= \min\|sR_0 - A\|_F^2\\
&= \min trace[(sR_0 - A)^T(sR_0  - A)]\\
\end{aligned}
$$

Introduce a symmetric lagrangian multiplier $$\Lambda = \Lambda^T$$
$$
\begin{aligned}
\hat{R}_0, \hat{\Lambda}, \hat{s} 
&= \min trace[(sR_0 - A)^T(sR_0  - A)] + trace[\Lambda(R_0^TR_0-I)]\\
\end{aligned}
$$

The derivative w.r.t. $$R_0, \Lambda, s$$ is 0, 

$$
\begin{aligned}
&2(sR_0 - A)s + R_0(\Lambda + \Lambda^T)=0 \Rightarrow s^2R_0 - As + R_0 \Lambda=0\\
&R_0^TR_0-I=0\\
&trace(2(sR_0-A)^T R_0)=0
\end{aligned}
$$

The solution to this objective:

$$
\begin{aligned}
&A = sR_0 + R_0 \Lambda\,/s= R_0(sI+\Lambda\,/s),\ R_0^TR_0-I=0 \Rightarrow A^TA = (sI+\Lambda\,/s)^2\\
&R_0 = A(sI+\Lambda\,/s)^{-1}=A(A^TA)^{-\frac{1}{2}}\\
&s = \frac{1}{2}trace(A^TR_0) = \frac{1}{2}trace((A^TA)^{\frac{1}{2}})\\
\end{aligned}
$$

Check $$\hat{R}_0 = A(A^TA)^{-\frac{1}{2}} = U_A\Sigma_AV_A^T(V_A\Sigma_A^2V_A^T)^{-\frac{1}{2}} = U_A\Sigma_AV_A^T(V_A\Sigma_A^{-1}V_A^T) = U_AV_A^T$$, $$\hat{s}=\frac{1}{2}trace((A^TA)^{\frac{1}{2}}) =\frac{1}{2}trace(V_A\Sigma_AV_A^T)= \frac{1}{2}trace(\Sigma_A)$$, same as the above solution derived from SVD.

## Implementation

We directly use OpenCV's function to do this. Below provides the function for 3 methods (general affine transformation, orthogonal transformation, post imposing orthogonal constraint) to align a face.


```
import numpy as np

def get_affine_matrix(orig_pts, target_pts):
    """
    input:
    orig_pts: N*2 original points location
    target_pts: N*2 target points location
    """
    N = orig_pts.shape[0]
    P = np.zeros((2 * N, 6))
    pa = target_pts.reshape((-1, 1), order='F')
    
    P[0:N, 0:2] = orig_pts
    P[0:N, 2] = np.ones((N, ))
    P[N:2*N, 3:5] = orig_pts
    P[N:2*N, 5] = np.ones((N, ))
    
    VA = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(P), P)),
                   np.transpose(P)), pa)
    return VA.reshape((2, 3))

def get_orthogonal_affine_matrix(orig_pts, target_pts):
    """
    input:
    orig_pts: N*2 original points location
    target_pts: N*2 target points location
    """
    N = orig_pts.shape[0]
    P = np.zeros((2 * N, 4))
    pa = target_pts.reshape((-1, 1), order='F')
    
    P[0:N, 0] = orig_pts[:, 0]
    P[0:N, 1] = -orig_pts[:, 1]
    P[0:N, 2] = np.ones((N, ))
    
    P[N:2 * N, 0] = orig_pts[:, 1]
    P[N:2 * N, 1] = orig_pts[:, 0]
    P[N:2 * N, 3] = np.ones((N, ))
    
    VA = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(P), P)),
                   np.transpose(P)), pa)
    A = np.zeros((2, 3))
    A[0:2, 0:2] = np.diag([VA[0,0], VA[0,0]]) + \
                    np.flip(np.diag([VA[1,0],-VA[1,0]]),axis=0)
    A[0:2, 2] = VA[2:4].reshape((-1))
    
    return A

def post_orthogonal_affine_matrix(A):
    """
    input:
    A: affine matrix
    """
    A22 = A[0:2,0:2]
    U, Sig, V = np.linalg.svd(A22,full_matrices=True)
    s = np.sum(Sig)/2.0
    A_orth = np.zeros(A.shape)
    A_orth[0:2,0:2] = s * np.matmul(U, np.transpose(V))
    A_orth[0:2, 2] = A[0:2, 2]
    return A_orth
```

### Face alignment result

The target points location are 
[[13.00 , 50.00 , 31.50],
[15.00 , 15.00 , 26.00 ]]

![face alignment using 3 methods](/assets/postimgs/How_to_align_a_face_1.png){:class="img-responsive"}

From the results we can see that:
1. Post orthogonal imposes ortho-constraint after getting the affine matrix, it is least accurate in the target landmark location.
2. General affine matrix introduces non-uniform scaling in different directions, may result in some warping of the faces.



