# Presentation

around 10 mins -- make it 15!

## Title

Hi, we are part ii students Hao Xiang Li and Qiaozhi Lei, and our project title is "Inverse problems with score modelling". 

## Introduction and Motivation (2mins)

<!-- [Introduce the problem. Explain why it is important.] -->

So what are inverse problems?

Consider a forward process where we apply measurements to images to get observations. Inverse problems are all about reversing forward processes: starting from observations, we seek to retrieve the original image. 

<!-- [footnote]: measurements have no noise -->

This is far from straightforward due to the ill-posed nature of many such problems. For instance, given a greyscale image, we can color it in different ways

We can see inverse problems as equivalent to conditional generation: we are trying to generate the original image, given the observation and the measurement as conditions.

We've focused on 5 specific inverse problems

Inpainting, where we fill in masked areas of images.
Colorisation, translating greyscale images into color.
Super-resolution, reverting the effects of a pooling kernel.
Deblurring, reverting a blurring kernel.
Class-conditional generation, where we generate images based on a label.

These inverse problems have real-world importances: colorization enables restorations of historical photographs for cultural preservation, while super-resolution in medical imaging allows for clearer and detailed diagnosis

<!-- [Explain how the main idea of the project relates to previous works and why it makes sense.] -->

The straightforward way to do conditional generation is through supervised learning, which demands extensive computational resources and retraining for each new condition.

Our approach, however, leverages the generalisability of an unconditional model -- specifically, score-based diffusion models

These models don't learn the entire distribution but instead the gradient of the log probability, known as the score. 

The forward diffusion process iteratively adds noise to an image. If we can reverse the foward process, we can get images from noise. The score guides the reverse process, according to this SDE.

<!-- equation of reverse SDE -->

To do conditional generation, we just need to modify the SDE slightly by making the log probability conditioned on the observation y. This allows us to use an unconditional model to solve various inverse problems without retraining for each condition -- we simply modify the inference step.

We employ the projector-corrector method to numerically solve our SDE, discretizing the process for practical computation.

<!-- [Explain some of the most relevant previous works.] -->

<!-- do we actually need this part? -->

Here are some relevant works, with more details later
1。 ysong, score sde, controllable generation 
2. ysong, medical, generalise on controllable generation
3. jiaming song, pseudo-inverse
4. MCG

## Method -- Baseline (2mins)

Let us introduce the Tweedie formula -- a powerful one step denoising method which is essential for some of the baseline algorithms.

Given a noisy image, if we can calculate its score and estimate the std of the noise, then we can use the tweedie formula to denoise the image in one step.

---

We can divide the baseline inference methods into 2 categories.

1. The first is gradient-based methods.

This is the SDE for conditional generation. 
<!-- shows equation -->
The gradient-based methods aim to estimate the gradient of the log probability. Using bayes rule, we get <!-- equation -->

Gradient-based approaches attempt to approximate this term. 
<!-- highlight the term on ppt  -->

For the task of class-conditional generation, we first train a noise-dependent classifier. We would softmax then log the classifier outputs to get the log probability, followed by backprogation to calculate its gradient.

Another approach is the Pseudoinverse Guided method.
- This approach approximates p(y|xt) with a normal distribution
<!-- shows distribution formula -->
- the gradient of the log probability can be represented as a vector-jacobian product
- In the equation, this term stands for the Moore-Penrose pseudoinverse, which can be considered as an approximated inverse of H.

---

2. The second is constraint-based methods

For each step in the reverse diffusion process, these methods add a corrective projection onto a constraint space to account for the condition y.

---

Consider image inpainting as an example. 
- Measurement H is a masking transformation
- y is the masked image

Corrective projection is to first get the ground truth sample by adding noise with the noise scale of the current step. Then we overwrite the known parts of the sample with the ground truth sample.

Intuitively, the ground truth samples in the known areas will guide the reverse diffusion process to generate something suitable in the unknown parts.

---

Colorisation is a special case of inapinting. At first, the known parts are coupled... 

If we use a decoupling matrix, we can decouple the greyscale data into its own channel, while the other two prevent data loss. The problem then becomes an inpainting problem where the first channel is known and the other 2 channels are unknown.

---

MCG combines the 2 approaches.
TODO: MCG

## Method -- Improvements (3mins)

We came up with a simple heuristic to estimate the term ...

It is easily tractable. Since p(xi|x) is a known gaussian, the result after passing through a linear function is still gaussian. 

<!-- show distribution -->

It is similar with the approximation of p(y|xt) in the pseudoinverse guided method. However, we didn't use tweedie for the mean. 

The results are surprisingly good on an initial subset of tasks.

<!-- show images of results -->

We conjecture that the performance of this heuristic could be due to this relationship for tasks where the score function doesn’t significantly impact the measurement

### Class-conditional 

The baseline algorithm uses a noise-conditional classifier, it would be really nice if we can use a noise-independent classifier, since we can use existing classifiers. 

But how to let a noise-independent classifier classify noisy samples? Our idea is using tweedie to denoise the noisy samples in one step, before passing to the classifier.

<!-- could show results here -->

### Problem Extensions

We applied many baseline methods to inverse problems not covered in the original papers. 

For super-resolution, our goal is to find HH^T to use the pseudoinverse guided method. 

We discover Aij to be ...

Therefore, HH^T can be found be be ... which is a diagonal matrix. This means the distribution is an i.i.d gaussian.

<!-- show distribution formula -->

Similar analysis can be applied to inpainting, where HHT is an identity matrix, and colorisation, which can be seen as super-resolution with can extra upscaling step.

Deblurring is a bit different. In deblurring, each element in the original image $x$ can influence multiple elements in the measured (blurred) $Hx$. As a result, $HH^T$ cannot be orthogonally transformed to a diagonal matrix, which means that $p_t(\bm{y}|\bm{x_i})$ can't be approximated as an i.i.d gaussian.

However, there is a clever way to find $H^+$ using SVD. 

## Method -- Data/ training/ testing (1min)

All our inferences require both a condition y and a measurement matrix H, sometimes derived from other information, such as the blurring kernel.

<!-- don't go into details here due to lack of time -->
<!-- footnote: the specific requirements for each task can be found in section 2.4.1 of our reports -->

We only need to train classifiers for the class conditional task.

<!-- [Explain input/ output data format, how it is consumed.] -->

## Results -- Datasets (1min)

<!-- [Explain the datasets utilized: what they contain, why they are utilized, assumptions, limitations, possible extensions.] -->

## Results -- Training and Testing (2min)

<!-- [Explain the training and testing results with graphs and elaborating on why they make sense, what could be improved.] -->

## Results -- Visual Results 

visual results shouldn't be a separate section in our presentation. we should show images of individual methods when we talk about them. 

## [Optional] Results -- Quantitative and Comparisons (2mins)

## Conclusions -- Main Takeaways (1min)

## Conclusions -- Limitations and Future Work (1min)
