import FileUploadForm from './FileUploadForm';

function SectionHero({ onUpload }) {
	return (
		<div className="grid items-center gap-6 lg:grid-cols-[1fr_500px] lg:gap-12 xl:grid-cols-[1fr_600px]">
			<div className="space-y-4">
				<h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl">
					Spice Up Your Style with FitGPT
				</h1>
				
				<p className="max-w-[600px] text-gray-400 md:text-xl">
					Our custom AI model analyzes pictures of people and recommends colors for their clothes.
					Try our fashion advice today!
				</p>
				
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
