package com.example.plant_classifier;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Point;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.plant_classifier.ml.PlantsModel;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;

import java.io.IOException;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    Button scan;
    TextView textDetected,moreDetails;
    ImageView capturedImg;
    private Bitmap imageBitmap;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        scan=findViewById(R.id.scanBtn);
        textDetected=findViewById(R.id.textDetected);
        capturedImg=findViewById(R.id.capturedImg);
        moreDetails=findViewById(R.id.moreDetails);

        scan.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                captureImage();
            }
        });
        moreDetails.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i=new Intent(Intent.ACTION_VIEW, Uri.parse("https://www.google.com/search?q="+textDetected.getText().toString()));
                startActivity(i);
            }
        });
    }

    static final int REQUEST_IMAGE_CAPTURE=1;
    private void captureImage(){
        Intent takePicture=new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if(takePicture.resolveActivity(getPackageManager())!=null){
            startActivityForResult(takePicture, REQUEST_IMAGE_CAPTURE);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode==REQUEST_IMAGE_CAPTURE && resultCode==RESULT_OK){
            Bundle extras =data.getExtras();
            imageBitmap=(Bitmap) extras.get("data");
            capturedImg.setImageBitmap(imageBitmap);
            outputGenerator(imageBitmap);
        }
    }
    private void outputGenerator(Bitmap imageBitmap){

        try {
            PlantsModel model = PlantsModel.newInstance(this);

            // Creates inputs for reference.
            TensorImage image = TensorImage.fromBitmap(imageBitmap);

            // Runs model inference and gets result.
            PlantsModel.Outputs outputs = model.process(image);
            List<Category> probability = outputs.getProbabilityAsCategoryList();
            int idx=0;
            float max=probability.get(0).getScore();
            for(int i=0;i<probability.size();i++){
                float curr_score=probability.get(i).getScore();
                if(max<curr_score){
                    max=curr_score;
                    idx=i;
                }
            }
            textDetected.setText(probability.get(idx).getLabel()+"\n"+probability.get(idx).getDisplayName());
            moreDetails.setText("View More Details");
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // 
        }
    }
}