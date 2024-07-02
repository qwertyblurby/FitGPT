import MultiBar from './MultiBar';
import UploadIcon from './assets/icons/UploadIcon';
import { useState } from 'react';
import SectionResults from './SectionResults';
import DemoData from './assets/demo/DemoData';

function SectionRecs() {
	const [results, setResults] = useState(null);
	const [uploadedImage, setUploadedImage] = useState(require("./assets/demo/sample.png"));
	const [demoStatus, setDemoStatus] = useState("sample"); // values: sample demo_1 demo_2 loading done
	
	const onUpload = async (event) => {
		console.log("call")
		try {
			setResults(null);
			setDemoStatus("loading");
			const fileInput = event.target;
			const file = fileInput.files[0];
			const fileURL = URL.createObjectURL(file);
			setUploadedImage(fileURL);
			if (file) {
				const formData = new FormData();
				formData.append('file', file);;
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
	}
	
	return (
		<div className="space-y-8">
			<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
				FitGPT Demo
			</h2>
			
			<div className="lg:flex gap-4 md:gap-8">
				{/* Demos */}
				<div className="flex gap-4">
					{/* Demo 1 */}
					<button
						className="btn relative w-40 h-60 rounded-lg overflow-hidden p-0"
						onClick={(e) => {updateStatus("demo_1")}}
					>
						<img
							src={require("./assets/demo/demo_1.jpg")}
							alt="Demo 1"
							className="object-cover w-full h-full"
						/>
					</button>

					{/* Demo 2 */}
					<button
						className="btn relative w-40 h-60 rounded-lg overflow-hidden p-0"
						onClick={(e) => {updateStatus("demo_2")}}
					>
						<img
							src={require("./assets/demo/demo_2.jpg")}
							alt="Demo 2"
							className="object-cover w-full h-full"
						/>
					</button>
					
					{/* Upload Button */}
					<div className="relative w-40 h-60">
						<form id="uploadForm" encType="multipart/form-data" className="size-full">
							<label
								htmlFor="fileInput"
								className="btn rounded-lg dark:bg-gray-800 dark:text-gray-50 dark:hover:bg-gray-700 size-full flex flex-col"
							>
								<p className="text-gray-300">Upload</p>
								<UploadIcon className="w-16 h-16 text-gray-300" />
							</label>
							<input
								type="file"
								name="file"
								id="fileInput"
								accept="image/*"
								className="hidden"
								aria-label="Upload your image"
								onChange={onUpload}
							/>
						</form>
					</div>
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
				
				<div className="bg-gray-800 rounded-lg p-6 flex-grow">
					<SectionResults results={results} uploadedImage={uploadedImage} demoStatus={demoStatus} />
				</div>
			</div>
		</div>
	)
}

export default SectionRecs;
