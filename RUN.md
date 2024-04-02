# VectorFusion Run Scripts

## Iconography

**baseline: text-to-image-to-vector:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 10 -respath ./workdir/ZPanda -d 8888
# set canvas size and image size:
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 10 -respath ./workdir/ZPanda -update "image_size=600" -d 8888
# rejection sampling -- K:
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 10 -respath ./workdir/ZPanda -update "K=20" -d 8888
# trainable stroke:
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 10 -respath ./workdir/stroke-ZPanda -update "train_stroke=True segment_init=random" -rdbz
# sd14, canvas size = 600, x_a size = 512
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 10 -respath ./workdir/sd14 -update "model_id=sd14 image_size=600" -rdbz -srange 100 200 
# sd15, canvas size = 600, x_a size = 512
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 10 -respath ./workdir/sd15 -update "model_id=sd15 image_size=600" -rdbz -srange 100 200
# sd21b, canvas size = 600, x_a size = 512
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 10 -respath ./workdir/sd21b -update "model_id=sd21b image_size=600" -rdbz -srange 100 200
# sd21, canvas size = 800, x_a size = 768
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 10 -respath ./workdir/sd21 -update "model_id=sd21 image_size=800 sds.im_size=768" -rdbz -srange 100 200
# sdxl, canvas size = 1100, x_a size = 1024
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 10 -respath ./workdir/sd21xl -update "model_id=sdxl image_size=1100 sds.im_size=1024" -rdbz -srange 100 200
```

**from scratch:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 10 -respath ./workdir/ZPanda -update "skip_live=True sds.num_iter=2000 path_reinit.stop_step=1500 path_reinit.area_threshold=0" -d 8888
```

**from scratch + SDXL:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 10 -respath ./VF/ZPanda -update "skip_live=True image_size=1024 sds.num_iter=2000 path_reinit.stop_step=1500 path_reinit.area_threshold=0 guidance_scale=5.0" -d 8888
```

## Sketch

**from scratch:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. minimal 2d line drawing. trending on artstation." -save_step 10 -respath ./workdir/ZPanda_sketch -update "style=sketch skip_live=True num_segments=5 radius=0.5" -d 8888
```

## Pixel-Art

**baseline: text-to-image-to-vector:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. pixel art. trending on artstation." -save_step 10 -respath ./workdir/ZPanda_pixel -update "style=pixelart" -d 8888
```
