import { useState } from 'react';
import DemoResults from './assets/demo/DemoResults';
import DemoData from './assets/demo/DemoData';
import DemoImage from './assets/demo/DemoImage';
import UploadButton from './assets/demo/UploadButton';

function SectionRecs() {
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
			if (file) {
				const formData = new FormData();
				formData.append('file', file);
				const response = await fetch('http://localhost:5000/upload', {
					method: 'POST',
					body: formData
				});
				
				if (!response.ok) {
					throw new Error("Network response was not ok");
				}
				
				const data = await response.json();
				setResults(data.output);
				setDemoStatus("done");
				setUploadedImage(fileURL);
				console.log("response received");
			} else {
				throw new Error("File not found");
			}
		} catch (error) {
			console.error("Error: ", error);
		};
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
				<div className="flex flex-col mt-5 lg:mt-2 space-y-4 lg:ml-4 ">
					<p className="text-gray-400 md:text-xl lg:text-lg">
						FitGPT provides suggestions by article of clothing, improving every aspect of an outfit. We give you a clean visual summary of the results.
					</p>
					<p className="text-gray-400 md:text-xl lg:text-lg">
						Try FitGPT now by uploading a picture of yourself, or select one of our demo pictures.
					</p>
				</div>
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

export default SectionRecs;
