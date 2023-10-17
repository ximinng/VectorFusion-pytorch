# Results

Here are some results.

## Iconography

### baseline: text-to-image-to-vector

- panda

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c vectorfusion.yaml \
-pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" \
-respath ./workdir/sd15/panda \
-update "model_id=sd15 image_size=600" \
-d 598522
```

You will get the following result:

![panda_iconography](img/panda_iconography.svg)

- boat

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c vectorfusion.yaml \
-pt "a boat. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" \
-respath ./workdir/sd15/boat \
-update "model_id=sd15 image_size=600" \
-d 79676
```

You will get the following result:

![boat_iconograghy](img/boat_iconograghy.svg)

- temple

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c vectorfusion.yaml \
-pt "A 3D rendering of a temple. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" \
-respath ./workdir/sd15/temple \
-update "model_id=sd15 image_size=600" \
-d 16025
```

You will get the following result:

![temple_iconography](img/temple_iconography.svg)

### from scratch

- panda

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c vectorfusion.yaml \
-pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" \
-respath ./workdir/sd15/panda \
-update "skip_live=True sds.num_iter=2000 path_reinit.stop_step=1500 path_reinit.area_threshold=0" \
-d 54091
```

You will get the following result:

![panda_icon_scratch](img/panda_icon_scratch.svg)

- boat

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c vectorfusion.yaml \
-pt "a boat. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" \
-respath ./workdir/sd15/boat \
-update "skip_live=True sds.num_iter=2000 path_reinit.stop_step=1500 path_reinit.area_threshold=0" \
-d 10630
```

You will get the following result:

![boat_icon_scratch](img/boat_icon_scratch.svg)

- temple

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c vectorfusion.yaml \
-pt "A 3D rendering of a temple. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" \
-respath ./workdir/sd15/temple \
-update "skip_live=True sds.num_iter=2000 path_reinit.stop_step=1500 path_reinit.area_threshold=0" \
-d 42
```

You will get the following result:

![temple_icon_scratch](img/temple_icon_scratch.svg)

## Pixel

- panda

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c vectorfusion.yaml \
-pt "a panda rowing a boat in a pond. pixel art. trending on artstation." \
-respath ./workdir/sd15/panda \
-update "style=pixelart image_size=512" \
-d 29067
```

You will get the following result:

![panda_pixel](img/panda_pixel.svg)

- boat

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c vectorfusion.yaml \
-pt "a boat. pixel art. trending on artstation." \
-respath ./workdir/sd15/boat \
-update "style=pixelart image_size=512" \
-d 52374
```

You will get the following result:

![boat_pixel](img/boat_pixel.svg)

- temple

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c vectorfusion.yaml \
-pt "A 3D rendering of a temple. pixel art. trending on artstation." \
-respath ./workdir/sd15/temple \
-update "style=pixelart image_size=512" \
-d 62559
```

You will get the following result:

![temple_pixel](img/temple_pixel.svg)

## Sketch

- panda

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c vectorfusion.yaml \
-pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. minimal 2d line drawing. trending on artstation." \
-respath ./workdir/sd15/panda \
-update "style=sketch skip_live=True num_segments=5 radius=0.5" \
-d 44629
```

You will get the following result:

![panda_sketch](img/panda_sketch.svg)

- boat

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c vectorfusion.yaml \
-pt "a boat. minimal flat 2d vector icon. minimal 2d line drawing. trending on artstation." \
-respath ./workdir/sd15/boat \
-update "style=sketch skip_live=True num_segments=5 radius=0.5" \
-d 62392
```

You will get the following result:

![boat_sketch](img/boat_sketch.svg)

- temple

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c vectorfusion.yaml \
-pt "A 3D rendering of a temple. minimal flat 2d vector icon. minimal 2d line drawing. trending on artstation." \
-respath ./workdir/sd15/temple \
-update "style=sketch skip_live=True num_segments=5 radius=0.5" \
-d 31002
```

You will get the following result:

![temple_sketch](img/temple_sketch.svg)