import FileUploadForm from './FileUploadForm';

function SectionHero({ onUpload }) {
	return (
		<div className="grid items-center gap-6 lg:grid-cols-[1fr_500px] lg:gap-12 xl:grid-cols-[1fr_600px]">
			<div className="space-y-4">
				<h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl">
					Unleash the Power of AI Image Classification
				</h1>
				
				<p className="max-w-[600px] text-gray-400 md:text-xl">
					Our cutting-edge AI model can accurately classify a wide range of images with lightning-fast speed.
					Experience the future of visual recognition today.
				</p>
				
				{/* <div className="flex items-center gap-4">
					<button className="btn dark:bg-gray-800 dark:text-gray-50 dark:hover:bg-gray-700">
						Upload Image
					</button>
					<input accept="image/*" aria-label="Upload image" className="hidden" type="file" />
				</div> */}
				<FileUploadForm onUpload={onUpload}/>
				
			</div>
			
			<img
				alt="Hero"
				className="mx-auto aspect-video overflow-hidden rounded-xl object-cover sm:w-full"
				height="400"
				src={require("./assets/coffindance.png")}
				width="600"
			/>
		</div>
	)
}

export default SectionHero;
