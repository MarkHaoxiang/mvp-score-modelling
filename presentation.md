# Presentation

around 10 mins -- make it 15!

## Title

Hi, we are two part II students taking the MVP module, and our project is "Inverse problems with score modelling". 

## Introduction and Motivation (3-4mins)

<!-- [Introduce the problem. Explain why it is important.] -->

So what are inverse problems?

Consider a forward process where we apply measurements to images to get observations. Inverse problems are all about reversing forward processes: starting from observations, it is desireable to retrieve the original image. 

<!-- [footnote]: measurements have no noise -->

This is far from straightforward due to the ill-posed nature of many such problems. For instance, given a greyscale image, we can color it in different ways

Inverse problems are equivalent to conditional generation in a way: we are trying to generate the original image, given the observation and the measurement as conditions.

We've focused on 5 specific inverse problems

Inpainting, filling in masked areas of images.
Colorisation, coloring greyscale images.
Super-resolution, reverting the effects of a pooling kernel.
Deblurring, reverting a blurring kernel.
Class-conditional generation, generating images based on a label.

Solving these inverse problems have real-world importances: colorization enables restorations of historical photographs, while super-resolution in medical imaging allows for clearer and detailed diagnosis

<!-- [Explain how the main idea of the project relates to previous works and why it makes sense.] -->

The straightforward way to do conditional generation is by retraining a diffusion model on a subset of data , which demands extensive computational resources and retraining for each new condition.

The approaches we reproduce, however, leverages the generalisability of an unconditional model -- specifically, score-based diffusion models

These models don't learn the entire distribution but instead the gradient of the log probability, known as the score. 


<!-- Page on forward diffusion -->

The forward diffusion process iteratively adds noise to an image. By reversing the forward process, it is possible to obtain images from noise. The score guides the reverse process, according to this Stochastic Differential Equation. There are many numerical methods for solving this, such as the Euler-Maruyama extension of Euler's method. In general, the continuous process is approximated through many small discrete steps.

<!-- equation of reverse SDE -->

To do conditional generation, modify the SDE slightly by making the log probability conditioned on the observation y. This allows us to use an unconditional model to solve various inverse problems without retraining for each condition, by modifying the inference step.

We employ the projector-corrector method by Song to numerically solve our SDE, discretizing the process for practical computation.

<!-- [Explain some of the most relevant previous works.] -->

<!-- Just have it as a single page on the PPT -->

Here are some relevant works to conditional generation which we reproduce and improve.
The first paper introduced score-sde models, gradient based class-conditional generation and constraint-based inpainting and colorisation methods. 
The second paper introduced the pseudoinverse guided method
The third paper introduced the manifold constrained gradient methods.

## Method -- Baseline (2mins)

<!-- Make sure to add citations -->
Introduce the Tweedie formula  -- a powerful one step denoising method which is essential for some of the baseline methods.

Given a noisy image, if we can calculate its score and estimate the std of the noise, then we can use the Tweedie formula to denoise the image in one step.

---

We can roughly divide the baseline inference methods into 2 categories.

1. The first are gradient-based methods.

This is the SDE for conditional generation. 
<!-- shows equation -->
The gradient-based methods aim to estimate the gradient of the log probability. Unfortunately, that is usually not tractable. Using bayes rule, however, we get 2 more tractable terms <!-- equation -->

For the task of class-conditional generation, we first train a noise-conditional classifier. We would softmax then log the classifier outputs to get the log probability, followed by backprogation to calculate its gradient.

Another approach is the Pseudoinverse Guided method.
- This approach approximates p(x_0 | x_t), and thus for linear kernels p(y|xi) through a normal distribution
<!-- shows distribution formula -->
- Assuming noiseless measurements, the gradient of the log probability can be represented as a vector-jacobian product
- In the equation, this term stands for the Moore-Penrose pseudoinverse, which can be considered as an approximated inverse of H.
- Intuitive "pseudoinverses" also work for non-linear measurements, such as JPEG decompression
<!-- show $h h^+ h (x) = h(x)$  -->
---

1. The second are constraint-based methods

For each step in the reverse diffusion process, these methods add a corrective projection onto a constraint space to account for the condition y.

---

Consider image inpainting as an example. 
- Measurement H is a masking transformation
- y is the masked image

Corrective projection is to first get the ground truth sample by adding noise with the noise scale of the current step. Then overwrite the known parts of the sample with the ground truth sample.

Intuitively, the ground truth samples in the known areas will guide the reverse diffusion process to generate something suitable in the unknown parts.

---

Colorisation is a special case of inpainting.

If we use a decoupling matrix, we can decouple the greyscale data into its own channel, while the other two prevent data loss. The problem then becomes an inpainting problem where the first channel is known and the other 2 channels are unknown.

This idea of transform -> subsampling -> replace with ground truth -> transform inverse is generalised in this paper ("Song: Inverse Medical"), and possible for linear measurements.

---

MCG adds a gradient step on constraint based approaches.

The authors analyse the topological space of diffusion with a local linear assumption. They show a gradient step (equation here!) is orthogonal to the manifold (add that nice figure from the paper), so it conditions the generalisation without causing issues in the diffusion convergence process.

## Method -- Improvements (4mins)

When it comes to improvements, we came up with a simple heuristic to estimate the gradient term ...

It is easily tractable. Since p(xi|x) is a known gaussian, the result after passing through a linear function is still gaussian. 

<!-- show distribution -->

It is similar with the approximation of p(y|xi) in the pseudoinverse guided method. However, we didn't use tweedie for the mean. 

The results are surprisingly good on an initial subset of tasks.

<!-- show images of results -->

TODO: double check this sentence, I got it from the report, I don't quite understand "this relationship..."
We conjecture that the performance could be due to this relationship for tasks where the score function doesn’t significantly impact the measurement. 

### Class-conditional 

For class-conditional generation, the baseline method uses a noise-conditional classifier, it would be really nice if we can use a noise-independent classifier, since we can use existing classifiers. 

But how to let a noise-independent classifier classify noisy samples? Our idea is using tweedie to denoise the noisy samples in one step, before passing to the classifier.

<!-- could show results here -->

### Problem Extensions

We applied many baseline methods to inverse problems not covered in the original papers. 

For super-resolution, our goal is to find HH^T to use the pseudoinverse guided method. 

First consider integer k, and similar `averaging'  transformation $A \in \left\{\frac{1}{k}, 0\right\}^{m \times n}$ of this form

<!-- show Aij definition -->

$A A^T$ is in fact $\frac{1}{k} \bm{I}$ of size $(m \times m)$. 

It is possible to decompose $H=PM$, where $M$ represents some sequence of elementary row reordering. In this case, $k$ is the area of each pooling block. Since $M$ is by definition orthogonal and invertible,

<!-- show HH^T derivation -->

HH^T can be found be be ... which is a diagonal matrix. This means the distribution is an i.i.d gaussian.

<!-- show distribution formula -->

Similar analysis can be applied to inpainting, where HHT is an identity matrix, and colorisation, which can be seen as super-resolution with can extra upscaling step.

TODO:
Quick slide on face merging?

Yes!

---

Deblurring is a bit different. In deblurring, each element in the original image $x$ can influence multiple elements in the blurred image $Hx$. As a result, $HH^T$ cannot be orthogonally transformed to a diagonal matrix, which means that $p_i(y|x_i)$ can't be approximated as an i.i.d gaussian.

However, we can try to find the moore penrose pseudoinverse, which can be found using the single value decomposition of H. 

To find SVD of H, we consider a 9 by 9 uniform blurring kernel, we can separate the blurring matrix H into A_r and A_c, which apply 1D bluring kernels on the rows and columns. 

<!-- show visualisations of H, Ar, Ac -->

Given that we can easily find the SVDs of A_r and A_c, we can write H in this way, which is the SVD of H after a sorting permuation.

Recall that moore penrose pseudoinverse are approximations of the actual inverse, and in many deblurring situations, applying the pesudoinverse directly on the observation yields very good results.


## Method -- Data/ training/ testing (1min)

All of our inferences require both a condition y and a measurement matrix H, sometimes derived from other information, such as the blurring kernel. The specific requirements for H and y are task-specific. For example, colorisation requires the condition to be 256 by 256 images, since the unconditional model is trained with images of such dimensions.

<!-- don't go into details here due to lack of time -->
<!-- footnote: the specific requirements for each task can be found in section 2.4.1 of our reports -->

We only need to train classifiers for the class conditional task. We preprocess pixel data into images of 3x256x256, scaled between 0 and 1.

We use 45% and 5% of the celeb dataset as training and testing datasets, while ensuring even distribution of male and female images in both datasets.

<!-- [Explain input/ output data format, how it is consumed.] -->

## Results -- Datasets (1min)

<!-- [Explain the datasets utilized: what they contain, why they are utilized, assumptions, limitations, possible extensions.] -->

Our first dataset chosen was CelebA-HQ, an annotated facial dataset. This was chosen for qualitative analysis, since humans are naturally good at detecting discrepancies on facial data. 

Our second dataset used was lsun-church, a outerchurch dataset with a wide range of architectural styles, lighting and background. This offers a good contrast with CelebA and the distribution of churches covers a much wider range than faces.

A limitation is that these datasets are still in a single domain: image generation. An extension we could not complete was extension to audio generation with MIDI due to computational resources. We think the expansion of domains would introduce new forms of measurements / problems not previously considered, thus potentially exposing innovate directions for guiding conditional generation.

<!-- Keep this short, time constraints -->

## Results -- Training and Testing (2min)

<!-- [Explain the training and testing results with graphs and elaborating on why they make sense, what could be improved.] -->

<!-- We really don't have that much training involved in this process. Let's move results under a single banner, both quantiative, training and qualitative.-->

## Results -- Visual Results 

<!-- visual results shouldn't be a separate section in our presentation. we should show images of individual methods when we talk about them. -->

## [Optional] Results -- Quantitative and Comparisons (2mins)

Here is the comparision between different methods for all 5 inverse problems on the church dataset.

and here is the same comparison on the celeb dataset.

In general, we find that no single method outperformed others in all tasks. Our PrYt heuristic performs well on super-resolution, but we observe black and white artifacts on block inpainting. Constraint pipeline was fast to execute and had decent performance on all tasks. On the super-resolution task, with our own constraint, it surpassed the performance of all other pipelines with the highest PSNR score. $\Pi$GDM did well on the colorisation task, and was second behind MCG for random inpainting from SSIM, LPIPS and Y measurement error score, as well as human view. Overall, MCG did well on all tasks and score highly for metrics in general as well. It performs remarkably well for inpainting, for example in the top row of random inpainting MCG is difficult to distinguish from ground truth. We conjecture the extra gradient term allows it to converge faster on top of the Constraint projection.

### Classifier Conditioning

We also briefly consider the results of the class-conditional generation for genders, which for simplicity of this investigation we pose as a binary classification problem. 

Both methods are clearly imperfect. The Tweedie-conditional generator seems to have less of an influence, keeping in line more with the unconditional model. Both methods successfully guide the model towards the correct class. There exists a limitation in both our method and the algorithm for noise-conditional classification: ie. we trained the two classifiers with different underlying architectures and weight balancing techniques. This could be improved on our part, but also reflects a key advantage of Tweedie classifier: it is far more easy to adapt/use existing work, such as the Mobilenet we used for the classifer, skipping the step of making it noise conditional.

<!-- skip  -->

## Conclusions -- Main Takeaways (1min)

- No current method solves all the posed problems (closest is MCG).
- Tweedie is a powerful tool to construct better estimates, allowing tweedie-conditional classifier, $\Pi$GDM and MCG to work.
- Tweedie methods are also slow for inference, due to backpropagation across the score model.
- Theoretical understanding of both the score model and measurements is key! Linear algebra.\
- Quantitative analysis is hard, better metrics are needed in both alignment and computational efficiency.

## Conclusions -- Limitations and Future Work (1min)
A major limitation of our work is on computational resources, which limit the confidence of our quantiative and qualititative analysis. Experiments we would have liked to run include:
1. Comparing FID scores between noise conditional and tweedie conditional class generation
2. More datapoints for qualitive results, such as face merging
3. Testing out how incorrect assumptions for $H$ degrade results (this was briefly explored for the task of pseudoinverse deblurring.)
Another limitation is that we consider perfect measurements without noise, and assume we have perfect knowledge of the measurement process, such as knowing the size of kernels.

On future work, it would be beneficial to observe how diffusion models extend to other domains with potentially new failure modes. We see potential and work needed in improving pipelines utilising neural networks. Finally, we are impressed by the performance of the MCG pipeline and similar gradient update steps deserve further exploration. 