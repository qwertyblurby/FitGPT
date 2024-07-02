import { useState } from 'react';
import DemoResults from './assets/demo/DemoResults';
import DemoData from './assets/demo/DemoData';
import DemoImage from './assets/demo/DemoImage';
import UploadButton from './assets/demo/UploadButton';

function SectionDemo() {
	const [results, setResults] = useState(DemoData("sample").results);
	const [uploadedImage, setUploadedImage] = useState(require("./assets/demo/sample.png"));
	const [demoStatus, setDemoStatus] = useState("sample"); // values: sample demo_1 demo_2 loading done
	
	const onUpload = async (event) => {
		try {
			setResults(null);
			setDemoStatus("loading");
			const fileInput = event.target;
			const file = fileInput.files[0];
			const fileURL = URL.createObjectURL(file);
			setUploadedImage(fileURL);

			const formData = new FormData();
			formData.append('file', file);

			// Create an AbortController to manage the timeout
			const controller = new AbortController();
			const timeoutId = setTimeout(() => {
				controller.abort();
			}, 20000); // 20 seconds timeout
			
			const response = await fetch('http://imaginarysiteherelmaodoesntexist.com:5000/upload', {
				method: 'POST',
				body: formData,
				signal: controller.signal,
			});

			clearTimeout(timeoutId); // Clear the timeout if request completes

			if (!response.ok) {
				throw new Error("Network response was not ok");
			}

			const data = await response.json();
			setResults(data.output);
			setDemoStatus("done");
			setUploadedImage(fileURL);
			console.log("response received");
		} catch (error) {
			console.error("Error: ", error);
			setDemoStatus("error");
		}
	};
	
	const updateStatus = (buttonId) => {
		setDemoStatus(buttonId);
		const data = DemoData(buttonId);
		setResults(data.results);
		setUploadedImage(data.imageSrc);
	};
	
	return (
		<div className="space-y-8">
			<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
				FitGPT Demo
			</h2>
			
			<div className="lg:flex gap-4 md:gap-8">
				{/* Demos */}
				<div className="flex gap-4">
					<DemoImage imageType="demo_1" updateStatus={updateStatus} />
					<DemoImage imageType="demo_2" updateStatus={updateStatus} />
					<UploadButton onUpload={onUpload} />
				</div>
				
				{/* Description */}
				<p className="mt-5 lg:mt-2 lg:ml-4 text-gray-400 md:text-xl lg:text-lg">
					FitGPT provides suggestions by article of clothing, improving every aspect of an outfit. We give you a clean visual summary of the results.
					<br /> <br />
					Try FitGPT now by uploading a picture of yourself, or select one of our demo pictures.
				</p>
			</div>
			
			<div className="gap-6 lg:flex lg:gap-12 pt-12">
				<img
					alt="Placeholder"
					className="overflow-hidden rounded-xl max-h-96"
					src={uploadedImage}
				/>
				<DemoResults results={results} demoStatus={demoStatus} />
			</div>
		</div>
	)
}

export default SectionDemo;