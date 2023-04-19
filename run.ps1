pushd $PSScriptRoot
    if (-not (Test-Path .\monodepth2)) {
        git clone https://github.com/nianticlabs/monodepth2.git monodepth2
        pushd monodepth2
            git checkout b676244e5a1ca55564eb5d16ab521a48f823af31
        popd
    }
    if (-not (Test-Path .\yolov4)) {
        git clone https://github.com/augmentedstartups/yolov4-custom-functions.git yolov4
        pushd yolov4
            git checkout 8f2c922a54d9ee0878efd3dffa42ce6e7bd3ae34
        popd
    }
    mv "./monodepth2-implementation.py" "./monodepth2/"
    mv "./yolov4-implementation.py"     "./yolov4/"
    python -m venv .venv
    Invoke-Expression ".\.venv\Scripts\activate"
    python -m pip install -r .\requirements.txt
    python main.py
popd