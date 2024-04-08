# Qualitative Results

## Iconography Style

### Case: Pandas

**Prompt:** a panda rowing a boat in a pond. <br/>
**Style:** iconography <br/>
**Preview:**

| <img src="./img/Icon-Pandas/sample.png" style="width: 250px; height: 250px;"> | <img src="./img/Icon-Pandas/live.svg" style="width: 250px; height: 250px;"> | <img src="./img/Icon-Pandas/finetune.svg" style="width: 250px; height: 250px;"> |
|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| (a) Sample raster image with Stable Diffusion                                 | (b) Convert raster image to a vector via LIVE                               | (c) VectorFusion: Fine tune by LSDS                                             |

**Script:**

```shell
python run_painterly_render.py -c vectorfusion.yaml -pt "a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 50 -respath ./workdir/pandas -d 294308
```

### Case: Starbucks

**Prompt:** A Starbucks coffee. <br/>
**Style:** iconography <br/>
**Preview:**

| <img src="./img/Icon-Starbucks/sample.png" style="width: 250px; height: 250px;"> | <img src="./img/Icon-Starbucks/live.svg" style="width: 250px; height: 250px;"> | <img src="./img/Icon-Starbucks/finetune.svg" style="width: 250px; height: 250px;"> |
|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| (a) Sample raster image with Stable Diffusion                                    | (b) Convert raster image to a vector via LIVE                                  | (c) VectorFusion: Fine tune by LSDS                                                |

**Script:**

```shell
python run_painterly_render.py -c vectorfusion.yaml -pt "A Starbucks coffee. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 50 -respath ./workdir/starbucks -d 367022
```

### Case: Sloth

**Prompt:** A smiling sloth wearing a leather jacket, a cowboy hat and a kilt. <br/>
**Style:** iconography <br/>
**Preview:**

| <img src="./img/Icon-sloth/sample.png" style="width: 250px; height: 250px;"> | <img src="./img/Icon-sloth/live.svg" style="width: 250px; height: 250px;"> | <img src="./img/Icon-sloth/finetune.svg" style="width: 250px; height: 250px;"> |
|------------------------------------------------------------------------------|----------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| (a) Sample raster image with Stable Diffusion                                | (b) Convert raster image to a vector via LIVE                              | (c) VectorFusion: Fine tune by LSDS                                            |

**Script:**

```shell
python run_painterly_render.py -c vectorfusion.yaml -pt "A smiling sloth wearing a leather jacket, a cowboy hat and a kilt. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 50 -respath ./workdir/sloth -d 280328 
```

### Case: Owl

**Prompt:** an owl standing on a wire. <br/>
**Style:** iconography <br/>
**Preview:**

| <img src="./img/Icon-owl/sample.png" style="width: 250px; height: 250px;"> | <img src="./img/Icon-owl/live.svg" style="width: 250px; height: 250px;"> | <img src="./img/Icon-owl/finetune.svg" style="width: 250px; height: 250px;"> |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------|------------------------------------------------------------------------------|
| (a) Sample raster image with Stable Diffusion                              | (b) Convert raster image to a vector via LIVE                            | (c) VectorFusion: Fine tune by LSDS                                          |

**Script:**

```shell
python run_painterly_render.py -c vectorfusion.yaml -pt "an owl standing on a wire. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 50 -respath ./workdir/owl -d 857581 
```

### Case: Train

**Prompt:** a train. <br/>
**Style:** iconography <br/>
**Preview:**

| <img src="./img/Icon-train/sample.png" style="width: 250px; height: 250px;"> | <img src="./img/Icon-train/live.svg" style="width: 250px; height: 250px;"> | <img src="./img/Icon-train/finetune.svg" style="width: 250px; height: 250px;"> |
|------------------------------------------------------------------------------|----------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| (a) Sample raster image with Stable Diffusion                                | (b) Convert raster image to a vector via LIVE                              | (c) VectorFusion: Fine tune by LSDS                                            |

**Script:**

```shell
python run_painterly_render.py -c vectorfusion.yaml -pt "a train. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 50 -respath ./workdir/train -d 857581 
```

### Case: Fire

**Prompt:** fire. <br/>
**Style:** iconography <br/>
**Preview:**

| <img src="./img/Icon-fire/sample.png" style="width: 250px; height: 250px;"> | <img src="./img/Icon-fire/live.svg" style="width: 250px; height: 250px;"> | <img src="./img/Icon-fire/finetune.svg" style="width: 250px; height: 250px;"> |
|-----------------------------------------------------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| (a) Sample raster image with Stable Diffusion                               | (b) Convert raster image to a vector via LIVE                             | (c) VectorFusion: Fine tune by LSDS                                           |

**Script:**

```shell
python run_painterly_render.py -c vectorfusion.yaml -pt "fire. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 50 -respath ./workdir/fire -d 1452 
```

---

## Pixel-Art Style

### Case: Guitar

**Prompt:** Electric guitar. <br/>
**Style:** Pixel-Art <br/>
**Preview:**

| <img src="./img/Pixel-Guitar-2/sample.png" style="width: 250px; height: 250px;"> | <img src="./img/Pixel-Guitar-2/live.svg" style="width: 250px; height: 250px;"> | <img src="./img/Pixel-Guitar-2/finetune.svg" style="width: 250px; height: 250px;"> |
|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| (a) Sample raster image with Stable Diffusion                                    | (b) Convert raster image to a vector via LIVE                                  | (c) VectorFusion: Fine tune by LSDS                                                |

**Script:**

```shell
python run_painterly_render.py -c vectorfusion.yaml -pt "Electric guitar. pixel art. trending on artstation" -save_step 50 -respath ./workdir/guitar -update "style=pixelart" -d 428484  
```

### Case: Hamburger

**Prompt:** A delicious hamburger. <br/>
**Style:** Pixel-Art <br/>
**Preview:**

| <img src="./img/Pixel-hamburger/sample.png" style="width: 250px; height: 250px;"> | <img src="./img/Pixel-hamburger/live.svg" style="width: 250px; height: 250px;"> | <img src="./img/Pixel-hamburger/finetune.svg" style="width: 250px; height: 250px;"> |
|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| (a) Sample raster image with Stable Diffusion                                     | (b) Convert raster image to a vector via LIVE                                   | (c) VectorFusion: Fine tune by LSDS                                                 |

**Script:**

```shell
python run_painterly_render.py -c vectorfusion.yaml -pt "A delicious hamburger. pixel art. trending on artstation" -save_step 50 -respath ./workdir/hamburger -update "style=pixelart" -d 499578
```

### Case: Pikachu

**Prompt:** Pikachu. <br/>
**Style:** Pixel-Art <br/>
**Preview:**

| <img src="./img/Pixel-Pikachu/sample.png" style="width: 250px; height: 250px;"> | <img src="./img/Pixel-Pikachu/live.svg" style="width: 250px; height: 250px;"> | <img src="./img/Pixel-Pikachu/finetune.svg" style="width: 250px; height: 250px;"> |
|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| (a) Sample raster image with Stable Diffusion                                   | (b) Convert raster image to a vector via LIVE                                 | (c) VectorFusion: Fine tune by LSDS                                               |

**Script:**

```shell
python run_painterly_render.py -c vectorfusion.yaml -pt "A delicious hamburger. pixel art. trending on artstation" -save_step 50 -respath ./workdir/hamburger -update "style=pixelart" -d 499578
```

---

## Sketch Style

### Case: Dragon

**Prompt:** watercolor painting of a firebreathing dragon. <br/>
**Style:** Sketch <br/>
**Preview:**

| <img src="./img/Sketch-Dragon-2/svg_iter0.svg"> | <img src="./img/Sketch-Dragon-2/svg_iter500.svg"> | <img src="./img/Sketch-Dragon-2/finetune.svg"> |
|-------------------------------------------------|---------------------------------------------------|------------------------------------------------|
| SVG initialization                              | VectorFusion fine-tune 500 step                   | VectorFusion fine-tune 1500 step               |

**Script:**

```shell
python run_painterly_render.py -c vectorfusion.yaml -pt "watercolor painting of a firebreathing dragon. minimal 2d line drawing. trending on artstation" -save_step 50 -respath ./workdir/dragon-sketch -update "style=sketch num_segments=5 radius=0.5 sds.num_iter=1500" -d 947593  
```

### Case: The Eiffel Tower

**Prompt:** The Eiffel Tower. <br/>
**Style:** Sketch <br/>
**Preview:**

| <img src="./img/Sketch-EiffelTower/svg_iter0.svg"> | <img src="./img/Sketch-EiffelTower/svg_iter500.svg"> | <img src="./img/Sketch-EiffelTower/finetune.svg"> |
|----------------------------------------------------|------------------------------------------------------|---------------------------------------------------|
| SVG initialization                                 | VectorFusion fine-tune 500 step                      | VectorFusion fine-tune 1500 step                  |

**Script:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py -c vectorfusion.yaml -pt "The Eiffel Tower. minimal 2d line drawing. trending on artstation" -save_step 50 -respath ./workdir/EiffelTower-sketch -update "style=sketch skip_live=True num_segments=5 radius=0.5 sds.num_iter=1500" -d 965058  
```

### Case: Temple

**Prompt:** A 3D rendering of a temple. <br/>
**Style:** Sketch <br/>
**Preview:**

| <img src="./img/Sketch-Temple/svg_iter0.svg"> | <img src="./img/Sketch-Temple/svg_iter500.svg"> | <img src="./img/Sketch-Temple/finetune.svg"> |
|-----------------------------------------------|-------------------------------------------------|----------------------------------------------|
| SVG initialization                            | VectorFusion fine-tune 500 step                 | VectorFusion fine-tune 1500 step             |

**Script:**

```shell
python run_painterly_render.py -c vectorfusion.yaml -pt "A 3D rendering of a temple. minimal 2d line drawing. trending on artstation" -save_step 50 -respath ./workdir/temple-sketch -update "style=sketch num_segments=5 radius=0.5 sds.num_iter=1500" -d 809385  
```