# Qualitative Results

## Iconography Style

### Case: Sloth

**Prompt:** A smiling sloth wearing a leather jacket, a cowboy hat and a kilt. <br/>
**Style:** iconography <br/>
**Preview:**

| <img src="./img/Icon-sloth/sample.png" style="width: 250px; height: 250px;"> | <img src="./img/Icon-sloth/live.svg" style="width: 250px; height: 250px;"> | <img src="./img/Icon-sloth/finetune.svg" style="width: 250px; height: 250px;"> |
|------------------------------------------------------------------------------|----------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| (a) Sample raster image with Stable Diffusion                                | (b) Convert raster image to a vector via LIVE                              | (c) VectorFusion: Fine tune by LSDS                                            |

**Script:**

```shell
python run_painterly_render.py -c vectorfusion.yaml -pt "A smiling sloth wearing a leather jacket, a cowboy hat and a kilt. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 50 -respath ./workdir/sloth -update "K=6" -d 280328 
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
python run_painterly_render.py -c vectorfusion.yaml -pt "an owl standing on a wire. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 50 -respath ./workdir/owl -update "K=6" -d 857581 
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
python run_painterly_render.py -c vectorfusion.yaml -pt "a train. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation" -save_step 50 -respath ./workdir/owl -update "K=6" -d 857581 
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

---

## Sketch Style

