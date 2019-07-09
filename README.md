## TODO

- [x] Gradient accumulation
- [x] Albumentations
- [ ] Leave-one-class-out cross-validation
- [ ] Better backbone
- [ ] freeze backbone for training to run faster
- [ ] crop objects from some examples and paste them to other images
- [ ] remove extra parts of the image
- [ ] class dropout, train "any-class" model `msh`
- [ ] label smoothing `msh` 
- [ ] check why new class prediction does not work: if it RPN or classification
- [ ] Reduce images, because large part of an image is not useful at all
- [ ] Try single-stage detectors
- [ ] Visualize predictions and train data (we would like to see what part of an image can be cropped out)
- [x] TensorboardX: training progress, learning rate, val metrics, etc
- [ ] Test-time augmentation
- [ ] Normal ensembling; ensembling models from previous epochs
- [ ] Add grayscale IR images to train procedure
