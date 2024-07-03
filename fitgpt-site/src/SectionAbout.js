import Profile from './Profile';

function SectionAbout() {
	const profiles = [
		[
			"berrybear06",
			<p>Interests: me, myself, and AI.</p>,
			"berrybearpfp.webp",
			"https://github.com/berrybear06"
		],
		[
			"qwertyblurby",
			<p>I'm Erik Shen</p>,
			"https://cdn.discordapp.com/attachments/624439411014369292/1257900780808044615/159072517.png?ex=668616e4&is=6684c564&hm=04d74439ef4fa9687dff572ee360f1fd6e9d6150e51581ee3e6ecbabcd0a1743&",
			"https://github.com/qwertyblurby"
		],
		[
			"chickeno7",
			<p>I know how to code on occasion.</p>,
			"https://cdn.discordapp.com/avatars/434530835962658817/c9b905a1828187809f6c94a91d9792ce.webp",
			"https://github.com/chicken07"
		],
		[
			"cronchbird",
			<p>I am eternal</p>,
			"https://avatars.githubusercontent.com/u/120995210?s=256&v=4",
			"https://github.com/cronchbird"
		],
	]
	return (
		<div className="space-y-6">
			<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">About Us</h2>
			<p className="text-gray-400">Meet the team behind FitGPT.</p>
			<p className="text-gray-400">It all started as a one-day hackathon project that barely worked, but we have come a long way since then.</p>
			<div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
				{profiles.map(([name, bio, imageSrc, ghSrc]) => (
					<Profile
						key={name}
						name={name}
						bio={bio}
						imageSrc={imageSrc}
						ghSrc={ghSrc}
					/>
				))}
			</div>
		</div>
	);
}

export default SectionAbout;
