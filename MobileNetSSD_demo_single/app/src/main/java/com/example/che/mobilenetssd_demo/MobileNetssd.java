package com.example.che.mobilenetssd_demo;

import android.graphics.Bitmap;

/**
 *  MobileNetssd的java接口，与本地c++代码相呼应 native为本地 此文件与 MobileNetssd.cpp相呼应
 */
public class MobileNetssd {

    public native boolean Init(byte[] param, byte[] bin); // 初始化函数
    public native float[] Detect(Bitmap bitmap); // 检测函数
    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("MobileNetssd");
    }
}

