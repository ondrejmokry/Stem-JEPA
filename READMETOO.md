Dataset musdb18

Stažení: https://sigsep.github.io/datasets/musdb.html

Dekódování: https://github.com/sigsep/sigsep-mus-io
    instalace docker desktop
    následně v konzoli WSL
    stažení: docker pull faroit/sigsep-mus-io
    spuštění: docker run --rm -v /mnt/d/Dokumenty/datasets/musdb18:/data faroit/sigsep-mus-io /scripts/decode.sh

Spuštění kódu

(envJEPA) ondrej@PC-066O5M1:/mnt/d/Dokumenty/GitHub/Stem-JEPA$ dora run data=my_data logger=csv trainer=gpu

Spuštění inference, kterou nabídl Claude Sonnet 4

(envJEPA) ondrej@PC-066O5M1:/mnt/d/Dokumenty/GitHub/Stem-JEPA$ python inference.py   --ckpt_path "/mnt/d/Dokumenty/GitHub/Stem-JEPA/logs/xps/b4bb67e6/checkpoints/last.ckpt"   --audio_path "/mnt/d/Dokumenty/datasets/musdb18-16kHz-singlefile/mixture.wav"