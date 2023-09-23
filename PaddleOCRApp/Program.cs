using OpenCvSharp;
using Sdcb.PaddleInference;
using Sdcb.PaddleOCR.Models.Local;
using Sdcb.PaddleOCR.Models;
using Sdcb.PaddleOCR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace PaddleOCRApp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var p = new Program();

            var t = Task.Run(() => p.Start());
            t.Wait();

            Console.ReadLine();
        }


        public async void Start()
        {
            FullOcrModel model = LocalFullModels.EnglishV4;


            byte[] sampleImageData;
            string sampleImageUrl = @"https://gatex.us.com/wp-content/uploads/2021/03/FCS-RP-Screen-600x452.jpg";
            using (HttpClient http = new HttpClient())
            {
                Console.WriteLine("Download sample image from: " + sampleImageUrl);
                sampleImageData = await http.GetByteArrayAsync(sampleImageUrl);
            }

            using (PaddleOcrAll all = new PaddleOcrAll(model, PaddleDevice.Mkldnn())
            {
                AllowRotateDetection = true, 
                Enable180Classification = false,
            })
            {
                // Load local file by following code:
                // using (Mat src2 = Cv2.ImRead(@"C:\test.jpg"))
                using (Mat src = Cv2.ImDecode(sampleImageData, ImreadModes.Color))
                {
                    PaddleOcrResult result = all.Run(src);
                    Console.WriteLine("Detected all texts: \n" + result.Text);
                    foreach (PaddleOcrResultRegion region in result.Regions)
                    {
                        Console.WriteLine($"Text: {region.Text}, Score: {region.Score}, RectCenter: {region.Rect.Center}, RectSize:    {region.Rect.Size}, Angle: {region.Rect.Angle}");
                    }
                }
            }
        }
    }
}
