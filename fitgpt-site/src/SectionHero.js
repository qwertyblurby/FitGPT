function SectionHero({ onUpload }) {
	return (
		<div className="grid items-center gap-6 lg:grid-cols-[1fr_400px] lg:gap-12 xl:grid-cols-[1fr_450px]">
			<div className="space-y-4">
				<h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl">
					AI-Powered Fashion with FitGPT
				</h1>
				
				<p className="max-w-[600px] text-gray-400 md:text-xl">
					Tired of stressing over what to wear? We're here to help.
					Get instant, personalized outfit suggestions to spice up your style!
				</p>
				
				<a href="#demo" className="btn dark:bg-gray-800 dark:text-gray-50 dark:hover:bg-gray-700">
					Upload Your Image
				</a>
			</div>
			
			<img
				alt="Hero"
				className="mx-auto aspect-video overflow-hidden rounded-xl object-cover sm:w-full"
				height="400"
				src={require("./assets/outfit_displays.jpg")}
				width="600"
			/>
		</div>
	)
}

export default SectionHero;
